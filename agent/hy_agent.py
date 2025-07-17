import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from torch.distributions import Categorical
import torch.nn.functional as F
from networks.network import HybridPolicy, ValueNet

LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95


class HybridAgent:
    def __init__(self, state_dim):
        self.policy_net = HybridPolicy(state_dim)
        self.value_net = ValueNet(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=LEARNING_RATE)
        self.gamma = GAMMA

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            output = self.policy_net(state)

        # 离散动作采样
        m1_dist = Categorical(logits=output["discrete_logits"]["model1"])
        m2_dist = Categorical(logits=output["discrete_logits"]["model2"])
        m1_action = m1_dist.sample()
        m2_action = m2_dist.sample()

        def get_continuous_action(mean, log_std):
            std = torch.exp(log_std).clamp(min=1e-6)
            dist = Normal(mean, std)
            x = dist.rsample()

            action = torch.tanh(x)
            action = (action + 1) / 2

            log_prob = dist.log_prob(x)
            log_prob -= torch.log(1 - torch.tanh(x).pow(2) + 1e-6)
            log_prob -= torch.log(torch.tensor(0.5))
            return action, log_prob

        sat_comp_action, sat_comp_logprob = get_continuous_action(
            output["continuous_params"]["sat_comp"][0],
            output["continuous_params"]["sat_comp"][1]
        )
        sat_bw_action, sat_bw_logprob = get_continuous_action(
            output["continuous_params"]["sat_bw"][0],
            output["continuous_params"]["sat_bw"][1]
        )
        ground_comp_action, ground_comp_logprob = get_continuous_action(
            output["continuous_params"]["ground_comp"][0],
            output["continuous_params"]["ground_comp"][1]
        )
        power_action, power_logprob = get_continuous_action(
            output["continuous_params"]["power"][0],
            output["continuous_params"]["power"][1]
        )

        total_logprob = (
                m1_dist.log_prob(m1_action) +
                m2_dist.log_prob(m2_action) +
                sat_comp_logprob +
                sat_bw_logprob +
                ground_comp_logprob +
                power_logprob
        ).squeeze()

        return {
            "discrete_actions": (m1_action.item(), m2_action.item()),
            "continuous_actions": (
                sat_comp_action.item(),
                sat_bw_action.item(),
                ground_comp_action.item(),
                power_action.item()
            ),
            "log_probs": total_logprob
        }

    def compute_advantages(self, rewards, states, next_states, dones):
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()

        advantages = []
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
            last_advantage = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_advantage
            advantages.insert(0, last_advantage)
        return torch.tensor(advantages)

    def compute_logprobs(self, outputs, actions):
        m1_dist = Categorical(logits=outputs["discrete_logits"]["model1"])
        m2_dist = Categorical(logits=outputs["discrete_logits"]["model2"])
        m1_actions = actions["discrete_actions"][0]
        m2_actions = actions["discrete_actions"][1]
        discrete_logprob = m1_dist.log_prob(m1_actions) + m2_dist.log_prob(m2_actions)

        def _get_cont_logprob(mean, log_std, action):
            scaled_action = action * 2 - 1
            x = torch.atanh(scaled_action.clamp(-0.999, 0.999))

            std = torch.exp(log_std)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(x)

            log_prob -= torch.log(1 - scaled_action.pow(2) + 1e-6)
            log_prob -= torch.log(torch.tensor(0.5))
            return log_prob

        cont_actions = actions["continuous_actions"]
        sat_comp_logprob = _get_cont_logprob(
            outputs["continuous_params"]["sat_comp"][0],
            outputs["continuous_params"]["sat_comp"][1],
            cont_actions[0]
        )
        sat_bw_logprob = _get_cont_logprob(
            outputs["continuous_params"]["sat_bw"][0],
            outputs["continuous_params"]["sat_bw"][1],
            cont_actions[1]
        )
        ground_comp_logprob = _get_cont_logprob(
            outputs["continuous_params"]["ground_comp"][0],
            outputs["continuous_params"]["ground_comp"][1],
            cont_actions[2]
        )
        power_logprob = _get_cont_logprob(
            outputs["continuous_params"]["power"][0],
            outputs["continuous_params"]["power"][1],
            cont_actions[3]
        )

        return discrete_logprob + sat_comp_logprob + sat_bw_logprob + ground_comp_logprob + power_logprob

    def compute_entropy(self, outputs):
        m1_dist = Categorical(logits=outputs["discrete_logits"]["model1"])
        m2_dist = Categorical(logits=outputs["discrete_logits"]["model2"])
        discrete_entropy = m1_dist.entropy() + m2_dist.entropy()

        def _get_cont_entropy(log_std):
            return 0.5 * (2 * torch.pi * torch.exp(2 * log_std) + 1).log()

        cont_entropy = sum([
            _get_cont_entropy(outputs["continuous_params"][k][1]).mean()
            for k in ["sat_comp", "sat_bw", "ground_comp", "power"]
        ])

        return (discrete_entropy + cont_entropy).mean()

    def update(self, batch):
        states = batch['states']
        next_states = batch['next_states']
        old_logprobs = batch['logprobs']
        rewards = batch['rewards']
        dones = batch['dones']

        with torch.no_grad():
            values = self.value_net(states).squeeze()
        advantages = self.compute_advantages(rewards, states, next_states,  dones)
        returns = advantages + values

        new_outputs = self.policy_net(states)
        new_logprobs = self.compute_logprobs(new_outputs, batch['actions'])

        ratio = (new_logprobs - old_logprobs).exp()
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 0.9, 1.1) * advantages
        ).mean()

        entropy = self.compute_entropy(new_outputs)
        total_policy_loss = policy_loss - 0.01 * entropy

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()

        new_values = self.value_net(batch['states']).squeeze()
        value_loss1 = F.mse_loss(new_values, returns)
        value_clipped = values + (new_values - values).clamp(-0.1, 0.1)
        value_loss2 = F.mse_loss(value_clipped, returns)
        value_loss = torch.max(value_loss1, value_loss2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()

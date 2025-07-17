import numpy as np
import torch
from collections import deque


class HybridReplayBuffer:
    def __init__(self, capacity=2048):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.logprobs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, discrete_action, continuous_action, logprob, reward, next_state, done):
        self.states.append(state.cpu().numpy() if torch.is_tensor(state) else state)
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.logprobs.append(logprob.item() if torch.is_tensor(logprob) else logprob)
        self.rewards.append(reward)
        self.next_states.append(next_state.cpu().numpy() if torch.is_tensor(next_state) else next_state)
        self.dones.append(done)

        if len(self.states) > self.capacity:
            self._remove_oldest()

    def _remove_oldest(self):
        self.states.pop(0)
        self.discrete_actions.pop(0)
        self.continuous_actions.pop(0)
        self.logprobs.pop(0)
        self.rewards.pop(0)
        self.next_states.pop(0)
        self.dones.pop(0)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.states), batch_size)

        batch = {
            'states': torch.FloatTensor(np.array([self.states[i] for i in indices])),
            'next_states':torch.FloatTensor(np.array([self.next_states[i] for i in indices])),
            'actions': {
                'discrete_actions': [
                    (self.discrete_actions[i][0], self.discrete_actions[i][1])
                    for i in indices
                ],
                'continuous_actions': [
                    (self.continuous_actions[i][0],
                     self.continuous_actions[i][1],
                     self.continuous_actions[i][2],
                     self.continuous_actions[i][3])
                    for i in indices
                ]
            },
            'logprobs': torch.stack([torch.tensor(self.logprobs[i]) for i in indices]),
            'rewards': torch.FloatTensor([self.rewards[i] for i in indices]),
            'dones': torch.FloatTensor([self.dones[i] for i in indices])
        }

        batch['actions']['discrete_actions'] = (
            torch.LongTensor([a[0] for a in batch['actions']['discrete_actions']]),
            torch.LongTensor([a[1] for a in batch['actions']['discrete_actions']])
        )

        batch['actions']['continuous_actions'] = (
            torch.FloatTensor([a[0] for a in batch['actions']['continuous_actions']]),
            torch.FloatTensor([a[1] for a in batch['actions']['continuous_actions']]),
            torch.FloatTensor([a[2] for a in batch['actions']['continuous_actions']]),
            torch.FloatTensor([a[3] for a in batch['actions']['continuous_actions']])

        )

        return batch

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states.clear()
        self.discrete_actions.clear()
        self.continuous_actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
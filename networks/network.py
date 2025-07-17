import torch
import torch.nn as nn


class HybridPolicy(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.discrete_base = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.continuous_base = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.model1_split_head = nn.Linear(128, 3)
        self.model2_split_head = nn.Linear(128, 3)

        for head in [self.model1_split_head, self.model2_split_head]:
            nn.init.orthogonal_(head.weight, gain=0.8)
            nn.init.constant_(head.bias, 0.1)

        self._build_improved_continuous_heads()

    def _build_improved_continuous_heads(self):
        hidden_dim = 128
        init_log_std = -1.0

        self.sat_comp_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        nn.init.constant_(self.sat_comp_head[-1].bias[0], 0.0)
        nn.init.constant_(self.sat_comp_head[-1].bias[1], init_log_std)

        self.sat_bw_head = self._build_single_continuous_head(init_log_std)
        self.ground_comp_head = self._build_single_continuous_head(init_log_std)
        self.power_head = self._build_single_continuous_head(init_log_std)

    def _build_single_continuous_head(self, init_log_std):
        head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        nn.init.orthogonal_(head[-1].weight, gain=1e-2)
        nn.init.constant_(head[-1].bias[0], 0.0)
        nn.init.constant_(head[-1].bias[1], init_log_std)
        return head

    def forward(self, x):
        discrete_feat = self.discrete_base(x)
        continuous_feat = self.continuous_base(x)
        def process_continuous(head, feat):
            params = head(feat)
            mean = torch.sigmoid(params[:, 0])
            log_std = params[:, 1].clamp(min=-3, max=1)
            return (mean, log_std)

        sat_comp = process_continuous(self.sat_comp_head, continuous_feat)
        sat_bw = process_continuous(self.sat_bw_head, continuous_feat)
        ground_comp = process_continuous(self.ground_comp_head, continuous_feat)
        power = process_continuous(self.power_head, continuous_feat)

        return {
            "discrete_logits": {
                "model1": self.model1_split_head(discrete_feat),
                "model2": self.model2_split_head(discrete_feat)
            },
            "continuous_params": {
                "sat_comp": sat_comp,
                "sat_bw": sat_bw,
                "ground_comp": ground_comp,
                "power": power
            }
        }


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )

        nn.init.orthogonal_(self.net[-1].weight, gain=0.1)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x):
        return self.net(x)
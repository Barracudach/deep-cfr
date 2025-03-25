
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .ops import DenseResidualBlock

class PolicyNetwork(nn.Module):
    def __init__(self, name, observation_space = 64, actions_amount = 5):
        super(PolicyNetwork, self).__init__()
        self.name = name
        self.observation_space = observation_space
        self.actions_amount = actions_amount

        self.encoder = nn.Sequential(
            DenseResidualBlock(self.observation_space, 128),
            DenseResidualBlock(128, 128),
            DenseResidualBlock(128, 128),
            DenseResidualBlock(128, 128),
            nn.Linear(128, self.actions_amount),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x, mask = x
        return self.encoder(x) * mask

    @torch.no_grad()
    def get_strategy(self, info_state:torch.Tensor, legal_actions_mask:torch.Tensor):

        # Expand dimensions to match batch size
        info_state = info_state.unsqueeze(0)  # Add batch dimension
        legal_actions_mask = legal_actions_mask.unsqueeze(0)  # Add batch dimension

        # Get advantages from the network
        strat = self.forward((info_state, legal_actions_mask))
        return strat.squeeze(0).cpu().numpy()

    def save(self):
        os.makedirs(f"./checkpoints", exist_ok=True)
        torch.save({'net': self.state_dict()}, f"./checkpoints/{self.name}.pth")

    def load(self):
        dat = torch.load(f"./checkpoints/{self.name}.pth")
        self.load_state_dict(dat['net'])


class AdvantageNetwork(nn.Module):
    def __init__(self, name, observation_space = 64, actions_amount = 5):
        super(AdvantageNetwork, self).__init__()
        self.name = name
        self.observation_space = observation_space
        self.actions_amount = actions_amount

        self.encoder = nn.Sequential(
            DenseResidualBlock(self.observation_space, 128),
            DenseResidualBlock(128, 128),
            DenseResidualBlock(128, 128),
            DenseResidualBlock(128, 128),
            nn.Linear(128, self.actions_amount)
        )

        nn.init.xavier_uniform_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        
    def forward(self, x):
        x, mask = x
        return self.encoder(x) * mask

    @torch.no_grad()
    def get_matched_regrets(self, info_state: torch.Tensor, legal_actions_mask: torch.Tensor, epsilon=1e-6):
        """
        Calculate regret matching.
        Args:
            info_state (torch.Tensor): Information state tensor.
            legal_actions_mask (torch.Tensor): Mask of legal actions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Advantages and matched regrets.
        """
        info_state = info_state.unsqueeze(0)
        legal_actions_mask = legal_actions_mask.unsqueeze(0)

        # Get advantages from the network
        raw_advantages = self.forward((info_state, legal_actions_mask))

        # Apply ReLU to get positive advantages
        advantages = torch.relu(raw_advantages)

        # Sum of positive advantages
        summed_regret = torch.sum(advantages, dim=1, keepdim=True)

        # Calculate matched regrets
        if summed_regret.item() > 0:
            matched_regrets = advantages / summed_regret
        elif summed_regret.item() ==0:
            matched_regrets = legal_actions_mask / legal_actions_mask.sum()
        else:
            # fallback: выбрать легальное действие с наибольшим "сырым" регретом
            masked_raw = raw_advantages.clone()
            masked_raw[legal_actions_mask == 0] = -float('inf')
            best_action = torch.argmax(masked_raw, dim=1, keepdim=True)

            matched_regrets = torch.zeros_like(raw_advantages)
            matched_regrets.scatter_(1, best_action, 1.0)

        return raw_advantages.squeeze(0).cpu().numpy(), matched_regrets.squeeze(0).cpu().numpy()
    
    def save(self):
        os.makedirs(f"./checkpoints", exist_ok=True)
        torch.save({'net': self.state_dict()}, f"./checkpoints/{self.name}.pth")

    def load(self):
        dat = torch.load(f"./checkpoints/{self.name}.pth")
        self.load_state_dict(dat['net'])
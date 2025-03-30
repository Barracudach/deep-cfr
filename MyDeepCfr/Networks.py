
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DenseResidualBlock(nn.Module):
    def __init__(self, input_dim, units):
        super(DenseResidualBlock, self).__init__()
        self.dense1 = nn.Linear(input_dim, units)
        self.dense2 = nn.Linear(units, units)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.layer_norm = nn.LayerNorm(units)

        if input_dim != units:
            self.shortcut = nn.Linear(input_dim, units)
        else:
            self.shortcut = nn.Identity()

        # Инициализация весов и смещений нулями
        nn.init.kaiming_uniform_(self.dense1.weight, a=0.1)
        nn.init.zeros_(self.dense1.bias)
        nn.init.kaiming_uniform_(self.dense2.weight, a=0.1)
        nn.init.zeros_(self.dense2.bias)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.kaiming_uniform_(self.shortcut.weight, a=0.1)
            nn.init.zeros_(self.shortcut.bias)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.dense1(x)
        x = self.leaky_relu(x)

        x = self.dense2(x)
        x = self.leaky_relu(x)

        x = x + shortcut
        x = self.leaky_relu(x)
        x = self.layer_norm(x)

        return x



class AdvantageNetwork(nn.Module):
    def __init__(self, name, observation_space = 64, actions_amount = 5):
        super(AdvantageNetwork, self).__init__()
        self.name = name
        self.observation_space = observation_space
        self.actions_amount = actions_amount
        
        self.encoder = nn.Sequential(
            nn.Linear(self.observation_space, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),

            nn.Linear(128, self.actions_amount)
            
        )

        # Initialize weights
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.1)
                nn.init.zeros_(m.bias)

        # self.encoder = nn.Sequential(
        #     DenseResidualBlock(self.observation_space, 128),
        #     DenseResidualBlock(128, 128),
        #     DenseResidualBlock(128, 128),
        #     DenseResidualBlock(128, 128),
        #     nn.Linear(128, self.actions_amount)
        # )

        nn.init.xavier_uniform_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        
    def forward(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def get_matched_regrets(self, info_state: torch.Tensor, legal_actions_mask: torch.Tensor, epsilon=1e-6):
        info_state = info_state.unsqueeze(0)
        legal_actions_mask = legal_actions_mask.unsqueeze(0)

        # Get advantages from the network
        raw_advantages = self.forward(info_state)
        
        mean_advantage = (raw_advantages.sum(dim=1, keepdim=True) /
                      legal_actions_mask.sum(dim=1, keepdim=True))

        raw_advantages = (raw_advantages - mean_advantage) * legal_actions_mask

        # Apply ReLU to get positive advantages
        advantages = torch.relu(raw_advantages)

        # Sum of positive advantages
        summed_regret = torch.sum(advantages, dim=1, keepdim=True)

        # Calculate matched regrets
        if summed_regret.item() > 0:
            matched_regrets = advantages / summed_regret
        else:
            # fallback: выбрать легальное действие с наибольшим "сырым" регретом
            masked_raw = raw_advantages.clone().masked_fill(legal_actions_mask == 0, -float('inf'))
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
        x = x
        return self.encoder(x)

    @torch.no_grad()
    def get_strategy(self, info_state:torch.Tensor, legal_actions_mask:torch.Tensor):

        # Expand dimensions to match batch size
        info_state = info_state.unsqueeze(0)  # Add batch dimension
        legal_actions_mask = legal_actions_mask.unsqueeze(0)  # Add batch dimension

        # Get advantages from the network
        strat = self.forward(info_state)
        return strat.squeeze(0).cpu().numpy()*legal_actions_mask.cpu().numpy()

    def save(self):
        os.makedirs(f"./checkpoints", exist_ok=True)
        torch.save({'net': self.state_dict()}, f"./checkpoints/{self.name}.pth")

    def load(self):
        dat = torch.load(f"./checkpoints/{self.name}.pth")
        self.load_state_dict(dat['net'])


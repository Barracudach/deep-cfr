
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class SkipDense(nn.Module):
    """Dense Layer with skip connection."""

    def __init__(self, units):
        super(SkipDense, self).__init__()
        self.hidden = nn.Linear(units, units)

    def forward(self, x):
        return self.hidden(x) + x


class PolicyNetwork(nn.Module):
    """Implements the policy network as an MLP with skip connections and layer normalization."""

    def __init__(self, input_size, policy_network_layers, num_actions, activation='leakyrelu'):
        super(PolicyNetwork, self).__init__()
        self._input_size = input_size
        self._num_actions = num_actions

        # Activation function
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = activation

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

        # Hidden layers
        self.hidden = nn.ModuleList()
        prev_units = self._input_size  # Используем input_size для первого слоя
        for units in policy_network_layers[:-1]:
            if prev_units == units:
                self.hidden.append(SkipDense(units))
            else:
                self.hidden.append(nn.Linear(prev_units, units))
            prev_units = units

        # Layer normalization
        self.normalization = nn.LayerNorm(prev_units)

        # Last hidden layer
        self.last_layer = nn.Linear(prev_units, policy_network_layers[-1])

        # Output layer
        self.out_layer = nn.Linear(policy_network_layers[-1], num_actions)

    def forward(self, inputs):
        """Applies Policy Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Action probabilities
        """
        x, mask = inputs

        # Apply hidden layers
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)

        # Apply layer normalization
        x = self.normalization(x)

        # Apply last hidden layer
        x = self.last_layer(x)
        x = self.activation(x)

        # Apply output layer
        x = self.out_layer(x)

        # Mask illegal actions
        x = torch.where(mask == 1, x, torch.tensor(-10e20, dtype=x.dtype, device=x.device))

        # Apply softmax
        x = self.softmax(x)

        return x
    
    @torch.no_grad()
    def get_strategy(self, info_state:torch.Tensor, legal_actions_mask:torch.Tensor):

        # Expand dimensions to match batch size
        info_state = info_state.unsqueeze(0)  # Add batch dimension
        legal_actions_mask = legal_actions_mask.unsqueeze(0)  # Add batch dimension

        # Get advantages from the network
        strat = self((info_state, legal_actions_mask))

        # Apply ReLU to get positive advantages
        strat = torch.relu(strat)

        return strat.squeeze(0).cpu().numpy()

class AdvantageNetwork(nn.Module):
    """Implements the advantage network as an MLP with skip connections and layer normalization."""

    def __init__(self, input_size, adv_network_layers, num_actions, activation='leakyrelu'):
        super(AdvantageNetwork, self).__init__()
        self._input_size = input_size
        self._num_actions = num_actions

        # Activation function
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = activation

        # Hidden layers
        self.hidden = nn.ModuleList()
        prev_units = self._input_size 
        for units in adv_network_layers[:-1]:
            if prev_units == units:
                self.hidden.append(SkipDense(units))
            else:
                self.hidden.append(nn.Linear(prev_units, units))
            prev_units = units

        # Layer normalization
        self.normalization = nn.LayerNorm(prev_units)

        # Last hidden layer
        self.last_layer = nn.Linear(prev_units, adv_network_layers[-1])

        # Output layer
        self.out_layer = nn.Linear(adv_network_layers[-1], num_actions)

    def forward(self, inputs):
        """Applies Advantage Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Cumulative regret for each info_state action
        """
        x, mask = inputs

        # Apply hidden layers
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)

        # Apply layer normalization
        x = self.normalization(x)

        # Apply last hidden layer
        x = self.last_layer(x)
        x = self.activation(x)

        # Apply output layer
        x = self.out_layer(x)

        # Mask illegal actions
        x = mask * x

        return x
    
    @torch.no_grad()
    def get_matched_regrets(self, info_state:torch.Tensor, legal_actions_mask:torch.Tensor):
        """
        Calculate regret matching.

        Args:
            info_state (torch.Tensor): Information state tensor.
            legal_actions_mask (torch.Tensor): Mask of legal actions.
            player (int): Player index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Advantages and matched regrets.
        """
        # Expand dimensions to match batch size
        info_state = info_state.unsqueeze(0)  # Add batch dimension
        legal_actions_mask = legal_actions_mask.unsqueeze(0)  # Add batch dimension

        # Get advantages from the network
        advs = self((info_state, legal_actions_mask))

        # Apply ReLU to get positive advantages
        advantages = torch.relu(advs)

        # Sum of positive advantages
        summed_regret = torch.sum(advantages, dim=1, keepdim=True)

        # Calculate matched regrets
        if summed_regret.item() > 0:
            matched_regrets = advantages / summed_regret
        else:
            # If all advantages are zero, choose the best legal action
            masked_advs = torch.where(legal_actions_mask == 1, advs, torch.full_like(advs, -1e20))
            best_action = torch.argmax(masked_advs, dim=1)
            matched_regrets = legal_actions_mask / legal_actions_mask.sum()

        return advantages.squeeze(0).cpu().numpy() , matched_regrets.squeeze(0).cpu().numpy() 
    
    @torch.no_grad()
    def get_matched_regrets2(self, info_states: torch.Tensor, legal_actions_masks: torch.Tensor, to_np=True):

        advantages_raw = self((info_states, legal_actions_masks))  # [batch_size, num_actions]
        advantages = F.relu(advantages_raw)

        sum_regret = advantages.sum(dim=0, keepdim=True)  # [batch_size, 1]
        sum_regret_expanded = sum_regret.expand_as(advantages)

        best_legal_deterministic = torch.zeros_like(advantages, dtype=torch.float32)
        best_actions = torch.argmax(
            torch.where(legal_actions_masks.bool(), advantages_raw, torch.full_like(advantages_raw, -1e20)),
            dim=0
        )
        batch_idx = torch.arange(info_states.size(0), device=info_states.device)
        best_legal_deterministic[ best_actions] = 1.0

        strategy = torch.where(sum_regret_expanded > 0,
                                advantages / sum_regret_expanded,
                                best_legal_deterministic )

        if to_np:
            return advantages.cpu().numpy(), strategy.cpu().numpy()
        return advantages, strategy
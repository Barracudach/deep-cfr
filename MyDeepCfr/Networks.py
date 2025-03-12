
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class ReservoirBuffer(object):
    """Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.

    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    """

    def __init__(self, reservoir_buffer_capacity):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        """Potentially adds `element` to the reservoir buffer.

        Args:
            element: data to be added to the reservoir buffer.
        """
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        """Returns `num_samples` uniformly sampled from the buffer.

        Args:
            num_samples: `int`, number of samples to draw.

        Returns:
            An iterable over `num_samples` random elements of the buffer.

        Raises:
            ValueError: If there are less than `num_samples` elements in the buffer
        """
        if len(self._data) < num_samples:
            raise ValueError('{} elements could not be sampled from size {}'.format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    @property
    def data(self):
        return self._data

    def shuffle_data(self):
        random.shuffle(self._data)


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
        prev_units = 0
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
        prev_units = 0
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
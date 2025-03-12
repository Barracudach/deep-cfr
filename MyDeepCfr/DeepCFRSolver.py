import os
import copy
from PokerRL.eval.head_to_head.H2HArgs import H2HArgs
from PokerRL.game.games import NoLimitHoldem,DiscretizedNLHoldem

import numpy as np
from PokerRL.game import Poker
from PokerRL.game._.tree._.nodes import PlayerActionNode
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.EvalAgentBase import EvalAgentBase as _EvalAgentBase
from PokerRL.rl.errors import UnknownModeError
import ray

import torch
import torch.nn as nn
import torch.optim as optim

from DeepCFR.workers.la.local import LearnerActor as LocalLearnerActor

from Networks import *

class DeepCFRSolver:
    def __init__(self,
               game,
               policy_network_layers=(256, 256),
               advantage_network_layers=(128, 128),
               num_iterations: int = 100,
               num_traversals: int = 100,
               learning_rate: float = 1e-3,
               batch_size_advantage: int = 2048,
               batch_size_strategy: int = 2048,
               memory_capacity: int = int(1e6),
               policy_network_train_steps: int = 5000,
               advantage_network_train_steps: int = 750,
               reinitialize_advantage_networks: bool = True,
               save_advantage_networks: str = None,
               save_strategy_memories: str = None,
               infer_device='cpu',
               train_device='cpu'):


        #TODO добавить сохранение 
        all_players = list(range(game.num_players()))
        self._game = game

        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._policy_network_train_steps = policy_network_train_steps
        self._advantage_network_train_steps = advantage_network_train_steps
        self._policy_network_layers = policy_network_layers
        self._advantage_network_layers = advantage_network_layers
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()
        self._embedding_size = len(self._root_node.information_state_tensor(0))
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._reinitialize_advantage_networks = reinitialize_advantage_networks
        self._num_actions = game.num_distinct_actions()
        self._iteration = 1
        self._learning_rate = learning_rate
        self._save_advantage_networks = save_advantage_networks
        self._save_strategy_memories = save_strategy_memories
        self._infer_device = infer_device
        self._train_device = train_device
        self._memories_tfrecordpath = None
        self._memories_tfrecordfile = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Launching on GPU")
        else:
            self.device = torch.device("cpu")
            print("Launching on CPU")
        
        # Initialize file save locations
        if self._save_advantage_networks:
            os.makedirs(self._save_advantage_networks, exist_ok=True)

        if self._save_strategy_memories:
            if os.path.isdir(self._save_strategy_memories):
                self._memories_tfrecordpath = os.path.join(
                    self._save_strategy_memories, 'strategy_memories.tfrecord')
            else:
                os.makedirs(
                    os.path.split(self._save_strategy_memories)[0], exist_ok=True)
                self._memories_tfrecordpath = self._save_strategy_memories

        # Initialize policy network, loss, optmizer
        self._reinitialize_policy_network()

        # Initialize advantage networks, losses, optmizers
        self._adv_networks = []
        self._adv_networks_train = []
        self._loss_advantages = []
        self._optimizer_advantages = []

        # Создание сетей и оптимизаторов для каждого игрока
        for player in range(self._num_players):
            # Создание основной сети
            self._adv_networks.append(
                AdvantageNetwork(self._embedding_size, self._advantage_network_layers,
                                self._num_actions).to(self.device)
            )

            # Создание сети для обучения (если требуется отдельная копия)
            self._adv_networks_train.append(
                AdvantageNetwork(self._embedding_size, self._advantage_network_layers,
                                self._num_actions).to(self.device)
            )

            # Функция потерь (Mean Squared Error)
            self._loss_advantages.append(nn.MSELoss())

            # Оптимизатор (Adam)
            self._optimizer_advantages.append(
                optim.Adam(self._adv_networks_train[player].parameters(), lr=learning_rate)
            )

        self._create_memories(memory_capacity)
    
    def _reinitialize_policy_network(self):
        self._policy_network = PolicyNetwork(self._embedding_size, 
                                             self._policy_network_layers,self._num_actions).to(self.device)
        self._optimizer_policy = optim.Adam(lr=self._learning_rate)
        self._loss_policy = nn.MSELoss()

    def _reinitialize_advantage_network(self, player):
        self._adv_networks_train[player] = AdvantageNetwork( self._embedding_size, 
                                                            self._advantage_network_layers, self._num_actions)
        self._optimizer_advantages[player] = optim.Adam(learning_rate=self._learning_rate)
        self._advantage_train_step[player] = (self._get_advantage_train_graph(player))

    
    def _advantage_train_step(self, player, info_states, advantages, iterations, masks, iteration):
        # Переводим данные в тензоры PyTorch
        info_states = torch.tensor(info_states, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        # Обнуляем градиенты
        self._optimizer_advantages[player].zero_grad()

        # Прямой проход
        preds = self._adv_networks_train[player]((info_states, masks))
        
        # Вычисление потерь
        sample_weight = iterations * 2 / iteration
        main_loss = self._loss_advantages[player](preds, advantages) * sample_weight

        # Обратный проход и обновление весов
        main_loss.backward()
        self.optimizer.step()

        return main_loss.item()
import os
import copy
from PokerRL.eval.head_to_head.H2HArgs import H2HArgs
from PokerRL.game.games import NoLimitHoldem,DiscretizedNLHoldem
import collections

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
from MyDeepCfr.EnvWrapper import EnvWrapper
from torch.utils.tensorboard import SummaryWriter

from .Networks import *
from .Datasets import *
import logging
import time


class DeepCFRSolver:
    def __init__(self,
               env_wrapper:EnvWrapper,
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
               reinitialize_advantage_networks: bool = False,
               save_advantage_networks: str = True,
               save_strategy_memories: str = True,
               adv_weight_decay=0.01,
               strat_weight_decay=0.01,
               enable_tb=True,
               infer_device='cpu',
               train_device='cpu'):
        
      
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        self._logger = logging.getLogger(__name__)
        
        env_wrapper.reset()
        self._env_wrapper:EnvWrapper = env_wrapper
        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._policy_network_train_steps = policy_network_train_steps
        self._advantage_network_train_steps = advantage_network_train_steps
        self._policy_network_layers = policy_network_layers
        self._advantage_network_layers = advantage_network_layers
        self._num_players = len(env_wrapper.env.seats)
        self._root_state = self._env_wrapper.state_dict()
        self._embedding_size = len(self._env_wrapper.get_current_obs())
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._reinitialize_advantage_networks = reinitialize_advantage_networks
        self._num_actions = env_wrapper.env.N_ACTIONS
        self._iteration = 1
        self._learning_rate = learning_rate
        self._save_advantage_networks = save_advantage_networks
        self._save_strategy_memories = save_strategy_memories
        self._infer_device = infer_device
        self._train_device = train_device
        self._memories_tfrecordpath = None
        self._memories_tfrecordfile = None
        self._adv_weight_decay= adv_weight_decay
        self._strat_weight_decay= strat_weight_decay
        self._enable_tb=enable_tb
            
        if self._enable_tb:
            self._tensorboard = SummaryWriter(f"./tensorboard/")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Launching on GPU")
        else:
            self.device = torch.device("cpu")
            print("Launching on CPU")
    

        # Initialize policy network, loss, optmizer
        self._reinitialize_policy_network()

        # Initialize advantage networks, losses, optmizers
        self._adv_networks:list[AdvantageNetwork] = []
        self._adv_networks_train:list[AdvantageNetwork] = []
        self._loss_advantages:list[nn.MSELoss] = []
        self._optimizer_advantages:list[optim.Adam]  = []

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
            self._loss_advantages.append(nn.MSELoss(reduction='none'))

            # Оптимизатор (Adam)
            self._optimizer_advantages.append(
                optim.Adam(self._adv_networks_train[player].parameters(), lr=learning_rate)
            )

        self._strategy_memories = StrategyDataset(memory_capacity)
        self._advantage_memories = [
            StrategyDataset(memory_capacity) for _ in range(self._num_players)
        ]

    def _reinitialize_policy_network(self):
        self._policy_network = PolicyNetwork(self._embedding_size, 
                                             self._policy_network_layers,self._num_actions).to(self.device)
        self._optimizer_policy = optim.Adam(params=self._policy_network.parameters(), 
                                            lr=self._learning_rate,
                                            weight_decay=self._strat_weight_decay)
        self._loss_policy = nn.MSELoss(reduction='none')

    def _reinitialize_advantage_network(self, player):
        self._adv_networks_train[player] = AdvantageNetwork( self._embedding_size, 
                                                            self._advantage_network_layers, self._num_actions)
        self._optimizer_advantages[player] = optim.Adam(params= self._adv_networks_train[player].parameters(),
                                                         lr=self._learning_rate, 
                                                         weight_decay=self._adv_weight_decay)
    
    def _advantage_train_step(self, player, info_states, advantages, iterations, masks, iteration):
        # Переводим данные в тензоры PyTorch
        info_states = torch.tensor(info_states, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

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
    


    def _traverse_game_tree(self, state, traverser, depth=0):

        legal_actions_mask =self._env_wrapper.legal_actions_mask()
        if state["current_player"] == traverser:
            _, strategy = self._adv_networks[traverser].get_matched_regrets(
                torch.tensor(self._env_wrapper.get_current_obs(), dtype=torch.float32).to(self.device),
                torch.tensor(legal_actions_mask, dtype=torch.float32).to(self.device))
            exp_payoff = np.zeros_like(strategy, dtype=np.float32)  # Инициализация exp_payoff
            for action in self._env_wrapper.legal_actions():
                _obs, _rew_for_all, _done, _info = self._env_wrapper.step(action)
                if _done:
                    exp_payoff[action] = _rew_for_all[traverser]
                else:
                    cur_state = self._env_wrapper.state_dict()
                    exp_payoff[action] = self._traverse_game_tree(cur_state, traverser, depth + 1)
                self._env_wrapper.load_state_dict(state)  # Восстановление состояния
            ev = np.sum(exp_payoff * strategy)
            samp_regret = (exp_payoff - ev) * legal_actions_mask
            self._advantage_memories[traverser].add(self._env_wrapper.get_current_obs(),self._iteration,samp_regret,legal_actions_mask)
            return ev
        else:
            _, strategy = self._adv_networks[state["current_player"]].get_matched_regrets(
                torch.tensor(self._env_wrapper.get_current_obs(), dtype=torch.float32).to(self.device),
                torch.tensor(legal_actions_mask, dtype=torch.float32).to(self.device))
            probs = strategy
            if probs.sum() > 0:
                probs /= probs.sum()
            else:
                probs = np.ones_like(probs) / len(probs)  # Равномерное распределение, если сумма равна нулю
            sampled_action = np.random.choice(range(self._num_actions), p=probs)
            self._strategy_memories.add(self._env_wrapper.get_current_obs(),self._iteration,strategy,legal_actions_mask)
            _obs, _rew_for_all, _done, _info = self._env_wrapper.step(sampled_action)
            if _done:
                return _rew_for_all[traverser]
            cur_state = self._env_wrapper.state_dict()
            ev = self._traverse_game_tree(cur_state, traverser, depth + 1)
            self._env_wrapper.load_state_dict(state)  # Восстановление состояния
            return ev
            
    def _learn_advantage_network(self, player):
        if len(self._advantage_memories[player]) < self._batch_size_advantage:
            return None

        dataloader = self._advantage_memories[player].get_dataloader(
            batch_size=self._batch_size_advantage,
            shuffle=True,
        )

        total_loss = 0.0
        for batch in dataloader:
            info_states, iterations, samp_regrets, legal_actions = batch

            info_states = info_states.to(self.device)
            samp_regrets = samp_regrets.to(self.device)
            legal_actions = legal_actions.to(self.device)

   
            model:AdvantageNetwork = self._adv_networks_train[player]
            optimizer:optim.Adam = self._optimizer_advantages[player]
            loss_func:nn.MSELoss=self._loss_advantages[player]

            optimizer.zero_grad()
            model.train()

            preds = model((info_states, legal_actions))
            sample_weight = (iterations * 2 / self._iteration).unsqueeze(1).to(self.device)
            main_loss:torch.Tensor = loss_func(preds, samp_regrets)*sample_weight
            main_loss=main_loss.mean()

            main_loss.backward()
            optimizer.step()
            total_loss += main_loss.sum()

        self._adv_networks[player].load_state_dict(
            self._adv_networks_train[player].state_dict()
        )

        return (total_loss / len(dataloader)).item()
    

    def _learn_strategy_network(self):
        if len(self._strategy_memories) < self._batch_size_advantage:
            return None

        dataloader = self._strategy_memories.get_dataloader(
            batch_size=self._batch_size_strategy,
            shuffle=True,
        )

        total_loss = 0.0
        for batch in dataloader:
            pass
    
    def load_checkpoint(self):
        try:
            data = torch.load(f'./checkpoints/{self._num_players}players.pt')

            self._policy_network.load_state_dict(data['policy_network'])
            for i in range(self._num_players):
                self._adv_networks[i].load_state_dict(data[f'adv_networks{i}'])

            self._players_steps = data['players_steps']
            self._iteration = data['iteration']
            self._logger.info("Checkpoint loaded")
        except Exception as e:
            self._logger.info(f"[-]Cant load checkpoint {self._num_players}players.pt {e}")

    def save_checkpoint(self):
        try:
            os.makedirs(f"./checkpoints", exist_ok=True)
            checkpoint = {
                'policy_network': self._policy_network.state_dict(),
                'players_steps': self._players_steps,
                'iteration': self._iteration,
            }

            for i in range(self._num_players):
                checkpoint[f'adv_networks{i}'] = self._adv_networks[i].state_dict()

            torch.save(checkpoint, f'./checkpoints/{self._num_players}players.pt')
            self._logger.info("Checkpoint saved")
        except Exception as e:
            self._logger.info(f"[-]Failed to save model {self._num_players}players.pt: {e}")

    def solve(self):
        self._players_steps=[0]*self._num_players
        self.load_checkpoint()
        
        for iter in range(self._num_iterations):
            self._logger.info(f"Iteration {iter}")
            for p in range(self._num_players):
                for traverse_iter in range(self._num_traversals):
                    self._logger.info(f"Traverse {p} player. Iter: {traverse_iter}/{self._num_traversals}")
                    start = time.time()
                    self._env_wrapper.reset()
                    self._traverse_game_tree(self._root_state, p)
                    self._logger.info(f"Traverse time:{time.time()-start}")

                if self._reinitialize_advantage_networks:
                    self._reinitialize_advantage_network(p)
                
                self._logger.info(f"Learn advantage network")
                loss=self._learn_advantage_network(p)

                if self._enable_tb:
                    self._tensorboard.add_scalar(f"loss/adv_net{p}", loss,self._players_steps[p])

                self._logger.info(f"Loss: {loss}")

                if self._save_advantage_networks :#and self._players_steps[p]%10==0
                    self.save_checkpoint()
                    
                self._players_steps[p]+=1

            self._iteration += 1

        self._logger.info("Learn strategy network")
        policy_loss = self._learn_strategy_network()

    #return self._policy_network, advantage_losses, policy_loss

    
import os


import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from MyDeepCfr.EnvWrapper import EnvWrapper
from torch.utils.tensorboard import SummaryWriter

from .Networks import *
from .Datasets import *
import logging
import time

from PokerEnv.games import DiscretizedNLHoldem

class DeepCFRSolver:
    def __init__(self,
               solver_idx,
               n_seats,
               bet_sizes_list_as_frac_of_pot,
               starting_stack_sizes_list,
               num_traversals: int = 100,
               learning_rate: float = 1e-3,
               batch_size_advantage: int = 2048,
               batch_size_strategy: int = 2048,
               memory_capacity: int = int(1e4),
               policy_epochs: int = 5,
               advantage_epochs: int = 5,
               reinitialize_advantage_networks: bool = False,
               save_strategy_memories: str = False,
               adv_weight_decay=0.01,
               strat_weight_decay=0.01,
               scale_rewards=False,
               enable_tb=True):
        
        self._solver_idx=solver_idx
        logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s - WORKER {self._solver_idx}] %(message)s')
        self._logger = logging.getLogger(__name__)

        env_args = DiscretizedNLHoldem.ARGS_CLS(n_seats=n_seats,
                                bet_sizes_list_as_frac_of_pot=bet_sizes_list_as_frac_of_pot,
                                stack_randomization_range=(0, 0,),
                                starting_stack_sizes_list=starting_stack_sizes_list,
                                scale_rewards=scale_rewards
                                )
        env= DiscretizedNLHoldem(env_args,lut_holder=DiscretizedNLHoldem.get_lut_holder(),is_evaluating=True)

        self._env_wrapper=EnvWrapper(env)
        self._env_wrapper.reset()
        
      
        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._policy_epochs = policy_epochs
        self._advantage_epochs = advantage_epochs
        self._num_players = len( self._env_wrapper.env.seats)
        self._embedding_size = len(self._env_wrapper.get_current_obs()["concat"])
        self._num_traversals = num_traversals
        self._reinitialize_advantage_networks = reinitialize_advantage_networks
        self._num_actions = self._env_wrapper.get_actions_count()
        self._iteration = 1
        self._learning_rate = learning_rate

        self._save_strategy_memories = save_strategy_memories
        self._adv_weight_decay= adv_weight_decay
        self._strat_weight_decay= strat_weight_decay
        self._enable_tb=enable_tb

        self._adv_net_name="adv_net"
        self._pol_net_name="pol_net"

        self._depth_after_which_one=10
        self._n_actions_traverser_samples=None

        matplotlib.use('Agg')
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
                AdvantageNetwork(f"adv_net{player}",self._embedding_size,
                                self._num_actions).to(self.device)
            )

            # Создание сети для обучения (если требуется отдельная копия)
            self._adv_networks_train.append(
                AdvantageNetwork(f"adv_net_train{player}",self._embedding_size,
                                self._num_actions).to(self.device)
            )

            self._loss_advantages.append(nn.MSELoss(reduction='none'))

            # Оптимизатор (Adam)
            self._optimizer_advantages.append(
                optim.Adam(self._adv_networks_train[player].parameters(), lr=learning_rate)
            )
        
        for i in range(1,self._num_players):
            self._adv_networks[i].load_state_dict(self._adv_networks[0].state_dict())

        self._strategy_memories = StrategyDataset(memory_capacity)
        self._advantage_memories = [
            AdvantageDataset(int(memory_capacity/2)) for _ in range(self._num_players)
        ]

        self._players_steps=[0]*self._num_players

        self._logger.info(f"Loading networks...")
        for i in range(self._num_players):
            self.load_network(self._adv_networks[i], self._adv_net_name+str(i))
            self._adv_networks_train[i].load_state_dict(
                self._adv_networks[i].state_dict()
            )   
        self.load_network(self._policy_network, self._pol_net_name)
        
        self._logger.info(f"Loading memory state...")
        self.load_memory_state()

        self._logger.info(f"Loading converge matrix...")

        self._converge_data=[]
        self._converge_data.append(np.load("./data/0;4,4;.npz")["tensor"])
        #self._converge_data.append(np.load("./data/1;4,4;2.npz")["tensor"])

    def get_model_weights(self):
        return {
            "policy_network": self._policy_network.state_dict(),
            "adv_networks": [net.state_dict() for net in self._adv_networks],
        }

    def set_model_weights(self, weights):
        self._policy_network.load_state_dict(weights["policy_network"])
        for i, net in enumerate(self._adv_networks):
            net.load_state_dict(weights["adv_networks"][i])

    def _reinitialize_policy_network(self):
        self._policy_network = PolicyNetwork("pol_net", self._embedding_size, self._num_actions).to(self.device)
        self._optimizer_policy = optim.Adam(params=self._policy_network.parameters(), 
                                            lr=self._learning_rate)
        self._loss_policy = nn.MSELoss(reduction='none')

    def _reinitialize_advantage_network(self, player):
        self._adv_networks_train[player] = AdvantageNetwork(f"adv_net_train{player}", self._embedding_size, self._num_actions).to(self.device)
        self._optimizer_advantages[player] = optim.Adam(params= self._adv_networks_train[player].parameters(),
                                                         lr=self._learning_rate)
        
    def _learn_advantage_network(self, player):

        dataloader = self._advantage_memories[player].get_dataloader(
            batch_size=self._batch_size_advantage,
            shuffle=True,
        )

        model: AdvantageNetwork = self._adv_networks_train[player]
        optimizer: optim.Adam = self._optimizer_advantages[player]
        loss_func: nn.MSELoss = self._loss_advantages[player]

        total_loss = 0.0
 
        for epoch in range(self._advantage_epochs):
            total_loss=0.0
            for batch in dataloader:
                info_states, iterations, samp_regrets, legal_actions = batch

                info_states = info_states.to(self.device)
                samp_regrets = samp_regrets.to(self.device)
                legal_actions = legal_actions.to(self.device)

                optimizer.zero_grad()
                model.train()

                preds = model((info_states, legal_actions))
                sample_weight = (iterations*2 / self._iteration).unsqueeze(1).to(self.device)

                main_loss: torch.Tensor = loss_func(preds, samp_regrets) * sample_weight
                main_loss = main_loss.mean()

                main_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += main_loss.item()
            

        self._adv_networks[player].load_state_dict(
            self._adv_networks_train[player].state_dict()
        )

        return total_loss / len(dataloader)

    def _learn_strategy_network(self):

        dataloader = self._strategy_memories.get_dataloader(
            batch_size=self._batch_size_strategy,
            shuffle=True
        )
        model:AdvantageNetwork = self._policy_network
        optimizer:optim.Adam = self._optimizer_policy
        loss_func:nn.MSELoss=self._loss_policy

        total_loss = 0.0
        for epoch in range(self._policy_epochs):
            total_loss=0.0
            for batch in dataloader:
                info_states, iterations, strategy, legal_actions = batch
                
                info_states = info_states.to(self.device)
                strategy = strategy.to(self.device)
                legal_actions = legal_actions.to(self.device)

            

                optimizer.zero_grad()
                model.train()

                preds = model((info_states, legal_actions))

                sample_weight = (iterations*2 / self._iteration).unsqueeze(1).to(self.device)
                main_loss:torch.Tensor = loss_func(preds, strategy)*sample_weight
                main_loss=main_loss.mean()

                main_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += main_loss.item()

        return total_loss / len(dataloader)


    def save_memory_state(self):
        for i in range(self._num_players):
            self._advantage_memories[i].save_buffer(f'./checkpoints/adv_buffer{i}.npz')
        self._strategy_memories.save_buffer(f'./checkpoints/pol_buffer.npz')
        np.savez_compressed(
            "./checkpoints/vars.npz",
            iteration=self._iteration,
        )

    def load_memory_state(self):
        try:
            for i in range(self._num_players):
                self._advantage_memories[i].load_buffer(f'./checkpoints/adv_buffer{i}.npz')
            self._strategy_memories.load_buffer(f'./checkpoints/pol_buffer.npz')
            data = np.load("./checkpoints/vars.npz")
            self._iteration=data["iteration"]
        except Exception as e:
            self._logger.info(f"[-]Cant load memory state. Exception:{e}")


    def save_all_networks(self):
        self.save_network(self._policy_network, self._pol_net_name)
        for i in range(self._num_players):
            self.save_network(self._adv_networks[i], self._adv_net_name+str(i))

    def save_network(self,network,name):
        os.makedirs(f"./checkpoints", exist_ok=True)
        torch.save({"net":network.state_dict()}, f'./checkpoints/{name}.pt')

    def load_network(self,network,name):
        try:
            data = torch.load(f'./checkpoints/{name}.pt')
            network.load_state_dict(data["net"])
        except Exception as e:
            self._logger.info(f"[-]Cant load network {name}.pt . Exception:{e}")
    

    def _traverse_game_tree(self, state, traverser, depth=0):
        legal_actions_mask = self._env_wrapper.legal_actions_mask()
        obs = self._env_wrapper.get_current_obs()
        current_player = state["current_player"]

        if current_player == traverser:
            _, strategy = self._adv_networks[traverser].get_matched_regrets(
                torch.tensor(obs["concat"], dtype=torch.float32).to(self.device),
                torch.tensor(legal_actions_mask, dtype=torch.float32).to(self.device))

            exp_payoff = np.zeros_like(strategy, dtype=np.float32)

            for i,action in enumerate(self._env_wrapper.legal_actions()):
                _obs, _rew_for_all, _done, _info = self._env_wrapper.step(action)
                
                if _done:
                    # if action==0:
                    #     exp_payoff[action]=0
                    # else:
                    exp_payoff[action] = _rew_for_all[traverser]
                else:
                    cur_state = self._env_wrapper.state_dict()
                    exp_payoff[action] = self._traverse_game_tree(cur_state, traverser, depth + 1)
                self._env_wrapper.load_state_dict(state)


            ev = np.sum(exp_payoff * strategy)
            samp_regret = (exp_payoff - ev) * legal_actions_mask
            
            self._advantage_memories[traverser].add(obs["concat"], int(self._iteration), samp_regret, legal_actions_mask)

            return ev

        else:
            _, strategy = self._adv_networks[current_player].get_matched_regrets(
                torch.tensor(obs["concat"], dtype=torch.float32).to(self.device),
                torch.tensor(legal_actions_mask, dtype=torch.float32).to(self.device))

            probs = strategy
            if probs.sum() > 0:
                probs /= probs.sum()
            else:
                probs = np.ones_like(probs) / len(probs)

            sampled_action = np.random.choice(range(self._num_actions), p=probs)
            self._strategy_memories.add(obs["concat"], int(self._iteration), probs, legal_actions_mask)

            _obs, _rew_for_all, _done, _info = self._env_wrapper.step(sampled_action)
            self._action_counts[traverser][sampled_action] += 1

            if _done:
                self._env_wrapper.load_state_dict(state)
                # if sampled_action==0:
                #     return 0
                # else:
                return _rew_for_all[traverser]

            cur_state = self._env_wrapper.state_dict()
            ev = self._traverse_game_tree(cur_state, traverser, depth + 1)
            self._env_wrapper.load_state_dict(state)
            return ev
    
    

    def PreflopGTOMatrixConverge(self,player_idx):
        if player_idx==0:
            stacks=[4,4]
            preflop_betline=[]
            board=""
            legal_actions_mask=[1,1,1]
        else:
            stacks=[4,4]
            preflop_betline=[2]
            board=""
            legal_actions_mask=[1,1,0]

        converge_matrix=np.zeros((13*13),dtype=np.float32)

        labels = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        for i, card1 in enumerate(labels):
            for j, card2 in enumerate(labels):
                if i < j:
                    hand = card1+'s' + card2+'s' 
                elif i >= j:
                    hand = card2 + 's' + card1 + 'd' 

                actions_ev=self._converge_data[player_idx][i* 13+ j] #call fold push

                #переделываем на наш формат (фолд чек/колл пуш)
                tmp=actions_ev[0]
                actions_ev[0]=actions_ev[1]
                actions_ev[1]=tmp

                obs=self._env_wrapper.eval_obs(stacks,preflop_betline,[],board,hand)
                
                _, strategy = self._adv_networks[player_idx].get_matched_regrets(
                    torch.tensor(obs["concat"], dtype=torch.float32).to(self.device),
                    torch.tensor(legal_actions_mask, dtype=torch.float32).to(self.device))
                
                net_ev = np.sum(actions_ev * strategy)
                solver_ev=np.max(actions_ev)
                delta=np.abs(net_ev-solver_ev)
                
                converge_matrix[i* 13+ j]=delta

        converge_matrix=converge_matrix.reshape((13,13))

        # Преобразуем в изображение: [C, H, W] — 1 канал, 13 высота, 13 ширина
        matrix_tensor = torch.tensor(converge_matrix, dtype=torch.float32).unsqueeze(0)  # -> [1, 13, 13]

        matrix_tensor /= (matrix_tensor.max() + 1e-8)

        self._tensorboard.add_image(f"converge/matrix/net{player_idx}", matrix_tensor, global_step=self._iteration)


    def solve(self):
        self._action_counts=[[0]*self._num_actions for _ in range(self._num_players)]

       
        for p in range(self._num_players):
            self._logger.info(f"Traverse {p} player")
            start = time.time()
            for traverse_iter in range(self._num_traversals):
                self._env_wrapper.reset()
                root_state = self._env_wrapper.state_dict()
                #print("_________________________________________")
                self._traverse_game_tree(root_state, traverser=p)

            if self._enable_tb:
                self._tensorboard.add_scalar(f"solver{self._solver_idx}/buffer/adv{p}", len(self._advantage_memories[p]) ,self._iteration)
                self._tensorboard.add_scalar(f"solver{self._solver_idx}/buffer/pol", len(self._strategy_memories) ,self._iteration)
            self._logger.info(f"Traverse time:{time.time()-start}")

            if len(self._advantage_memories[p]) < self._batch_size_advantage:
                continue
            
            if self._reinitialize_advantage_networks:
                self._reinitialize_advantage_network(p)
            
            self._logger.info(f"Learn advantage network")
            loss=self._learn_advantage_network(p)
            if self._enable_tb:
                self._tensorboard.add_scalar(f"solver{self._solver_idx}/loss/adv_net{p}", loss,self._iteration)
            self._logger.info(f"Loss: {loss}")
        
        for i in range(self._num_players):
            fig, ax = plt.subplots()
            bars = ax.bar(range(len(self._action_counts[i])), self._action_counts[i])
            ax.set_xlabel('Action')
            ax.set_ylabel('Counts')
            ax.set_title(f'Actions Distribution (solver {self._solver_idx}, net{i})')

            self._tensorboard.add_figure(
                f"solver{self._solver_idx}/actions/net{i}",
                fig,
                self._iteration
            )
            if i==0:
                self.PreflopGTOMatrixConverge(i)


        self._iteration += 1
        
        for i in range(self._num_players):
            if len(self._advantage_memories[i]) < self._batch_size_advantage:
                return
        if len(self._strategy_memories) < self._batch_size_strategy:
            return
        self._logger.info("Learn strategy network")
        policy_loss = self._learn_strategy_network()
        if self._enable_tb:
            self._tensorboard.add_scalar(f"solver{self._solver_idx}/loss/pol_net", policy_loss,self._iteration)
     
       

        

    #return self._policy_network, advantage_losses, policy_loss

    


@ray.remote(num_cpus=0.2, num_gpus=0.2 if torch.cuda.is_available() else 0)
class DistDeepCFRSolver(DeepCFRSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

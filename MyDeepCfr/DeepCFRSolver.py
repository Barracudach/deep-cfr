import os


import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict 
import hashlib

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
               adv_memory_capacity: int = int(1e5),
               strat_memory_capacity: int = int(1e4),
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

        self._strategy_memories = StrategyDataset(strat_memory_capacity)
        self._advantage_memories = [
            AdvantageDataset(adv_memory_capacity) for _ in range(self._num_players)
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

        self._regret_table = [defaultdict(lambda: np.zeros(self._num_actions, dtype=np.float32)) for _ in range(self._num_players)]


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

    def soft_update(self,target_model, source_model, tau):
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

    
    def _learn_advantage_network(self, player):

        info_states, iterations, samp_regrets, legal_actions = self._advantage_memories[player].get_data()

        model: AdvantageNetwork = self._adv_networks_train[player]
        optimizer: optim.Adam = self._optimizer_advantages[player]
        loss_func: nn.MSELoss = self._loss_advantages[player]


     
        total_loss = 0.0

        dataset_size = len(info_states)
        indices = np.arange(dataset_size)
        for epoch in range(self._advantage_epochs):
            total_loss = 0.0
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, self._batch_size_advantage):
                end_idx = min(start_idx + self._batch_size_advantage, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_info_states = torch.from_numpy(info_states[batch_indices]).to(self.device)
                batch_iterations = torch.from_numpy(iterations[batch_indices]).to(self.device)
                batch_samp_regrets = torch.from_numpy(samp_regrets[batch_indices]).to(self.device)
                batch_legal_actions = torch.from_numpy(legal_actions[batch_indices]).to(self.device)
                cur_iter = torch.tensor(self._iteration, dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                model.train()

                preds = model(batch_info_states) * batch_legal_actions
                sample_weight = (batch_iterations / cur_iter).unsqueeze(1).to(self.device)

                main_loss: torch.Tensor = loss_func(preds, batch_samp_regrets) * sample_weight
                main_loss = main_loss.mean()

                main_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += main_loss.item()

        self.soft_update(self._adv_networks[player], self._adv_networks_train[player], 0.4)

        return  total_loss /  (dataset_size / self._batch_size_advantage)

    def _learn_strategy_network(self):

        info_states, iterations, strategies, legal_actions = self._strategy_memories.get_data()
 
        model:AdvantageNetwork = self._policy_network
        optimizer:optim.Adam = self._optimizer_policy
        loss_func:nn.MSELoss=self._loss_policy

        total_loss = 0.0

        dataset_size = len(info_states)
        indices = np.arange(dataset_size)
        for epoch in range(self._policy_epochs):
            total_loss=0.0
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, self._batch_size_advantage):
                end_idx = min(start_idx + self._batch_size_advantage, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                    # собираем батч
                batch_info_states = torch.from_numpy(info_states[batch_indices]).to(self.device)
                batch_iterations = torch.from_numpy(iterations[batch_indices]).to(self.device)
                batch_strategies = torch.from_numpy(strategies[batch_indices]).to(self.device)
                batch_legal_actions = torch.from_numpy(legal_actions[batch_indices]).to(self.device)
                cur_iter=torch.tensor(self._iteration, dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                model.train()

                preds = model(batch_info_states)*batch_legal_actions

                sample_weight = (batch_iterations / cur_iter).unsqueeze(1).to(self.device)
                main_loss:torch.Tensor = loss_func(preds, batch_strategies)*sample_weight
                main_loss=main_loss.mean()

                main_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += main_loss.item()

        return total_loss /  (dataset_size / self._batch_size_advantage)


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
                
                #strategy=self.get_strategy_from_table(player_idx,legal_actions_mask,obs["concat"])
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

    def hash_info_state(self,info_state: np.ndarray) -> str:
        info_bytes = info_state.astype(np.float32).tobytes()
        return hashlib.sha256(info_bytes).hexdigest()
    
    def get_strategy_from_table(self,player,_legal_action_mask, info_state: np.ndarray):
        hash=self.hash_info_state(info_state)

        regrets=np.zeros(self._num_actions)
        if hash in self._regret_table[player]:
            regrets=self._regret_table[player][hash]

        advantages = np.maximum(0, regrets)
        summed_regret = np.sum(advantages)

        if summed_regret > 0:
            probabilities = advantages / summed_regret
        elif summed_regret==0:
            probabilities = np.ones_like(regrets) / len(regrets)  # равномерное распределение
        else:
            # fallback: выбрать единственное наибольшее по "сырым" регретам действие
            masked = regrets.copy()
            masked[self._legal_action_mask == 0] = -np.inf
            best_action = np.argmax(masked)
            probabilities = np.zeros_like(regrets)
            probabilities[best_action] = 1.0

        return probabilities
    
    def _traverse_game_tree(self, state, traverser, depth=0):
        legal_actions = self._env_wrapper.legal_actions()
        legal_actions_mask = np.zeros(self._env_wrapper.get_actions_count(), dtype=np.float32)
        legal_actions_mask[legal_actions]=1
        obs = self._env_wrapper.get_current_obs()
        current_player = state["current_player"]

        if current_player == traverser:

            #strategy=self.get_strategy_from_table(current_player,legal_actions_mask,obs["concat"])
            _, strategy = self._adv_networks[traverser].get_matched_regrets(
                torch.tensor(obs["concat"], dtype=torch.float32).to(self.device),
                torch.tensor(legal_actions_mask, dtype=torch.float32).to(self.device))

            exp_payoff = np.zeros_like(strategy, dtype=np.float32)

            for action in legal_actions:
                _obs, _rew_for_all, _done, _info = self._env_wrapper.step(action)
                
                if _done:
                    # if action==0:
                    #     exp_payoff[action]=0
                    # else:
                    exp_payoff[action] = _rew_for_all[traverser]
                else:
                    cur_state = self._env_wrapper.state_dict()
                    #new_obs = self._env_wrapper.get_current_obs()
                    #print(f"{obs['len1']}-{obs['where']}-{obs['redrawned_self_cards']}  |  {new_obs['len1']}-{new_obs['where']}-{new_obs['redrawned_self_cards']}")
                    exp_payoff[action] = self._traverse_game_tree(cur_state, traverser, depth + 1)
                self._env_wrapper.load_state_dict(state)

            ev = np.sum(exp_payoff * strategy)
            
            
            samp_regret = (exp_payoff - ev) * legal_actions_mask 
            #samp_regret = np.clip(samp_regret, -5, 5)

            # max_scaling_factor = 50.0
            # min_scaling_factor = 5.0

            # Плавно уменьшаем фактор сглаживания по мере увеличения итераций
            # regret_scaling_factor = max(
            #     min_scaling_factor, 
            #     max_scaling_factor / np.sqrt(self._iteration)
            # )
            # samp_regret = np.tanh(samp_regret / regret_scaling_factor) * regret_scaling_factor

            #print(f"raw_regrets:{_} samp_regrets: {samp_regret}")
            #max_abs = max(np.max(np.abs(samp_regret)), 1.0)
            #samp_regret = np.clip(samp_regret / max_abs, -1.0, 1.0)
            #samp_regret=np.maximum(samp_regret,0)

            self._action_counts[traverser][traverser][np.argmax(strategy)] += 1
            self._summ_regrets[current_player]+=samp_regret
            self._count_regrets[current_player]+=1

            #self._regret_table[traverser][self.hash_info_state(obs["concat"])]=samp_regret
            #if  samp_regret.sum()>1e-4:
            self._advantage_memories[traverser].add(obs["concat"], int(self._iteration), samp_regret, legal_actions_mask)

            return ev

        else:

            #strategy=self.get_strategy_from_table(current_player,legal_actions_mask,obs["concat"])
            _, strategy = self._adv_networks[current_player].get_matched_regrets(
                torch.tensor(obs["concat"], dtype=torch.float32).to(self.device),
                torch.tensor(legal_actions_mask, dtype=torch.float32).to(self.device))
            
            self._action_counts[traverser][current_player][np.argmax(strategy)] += 1

            self._strategy_memories.add(obs["concat"], int(self._iteration), strategy, legal_actions_mask)
            sampled_action = np.random.choice(range(self._num_actions), p=strategy)
        
            
            # if current_player==0:
            #     if legal_actions_mask[2]!=0:
            #         sampled_action=2

            _obs, _rew_for_all, _done, _info = self._env_wrapper.step(sampled_action)
            self._action_counts[traverser][current_player][sampled_action] += 1

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
    
    


    def solve(self):
        self._action_counts=[[[0]*self._num_actions for _ in range(self._num_players)]for _ in range(self._num_players)]
        self._summ_regrets=[[0 for __ in range(self._num_actions)] for _ in range(self._num_players)]
        self._count_regrets=[0 for _ in range(self._num_players)]

        for p in range(self._num_players):
            self._logger.info(f"Traverse {p} player")
            start = time.time()
            for traverse_iter in range(self._num_traversals):
                self._env_wrapper.reset()
                root_state = self._env_wrapper.state_dict()

                #self._env_wrapper.SetPlayersHands(["AsAd", "7s2d"]) #DEBUG

                self._traverse_game_tree(root_state,p)

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
            
            

     
        for net_idx in range(self._num_players):
            for player_idx in range(self._num_players):
                #actions
                fig, ax = plt.subplots()
                bars = ax.bar(range(len(self._action_counts[net_idx][player_idx])), self._action_counts[net_idx][player_idx])
                ax.set_xlabel('Action')
                ax.set_ylabel('Counts')
                ax.set_title(f'Actions Distribution net{net_idx} player{player_idx}')

                self._tensorboard.add_figure(
                    f"solver{self._solver_idx}/actions/net{net_idx}/player{player_idx}",
                    fig,
                    self._iteration
                )

            avg_regrets=self._summ_regrets[player_idx]/self._count_regrets[player_idx]

            for i, val in enumerate(avg_regrets):
                self._tensorboard.add_scalar(f"avg_regrets/net{net_idx}/action_{i}", val, self._iteration)


            if net_idx==0:
                self.PreflopGTOMatrixConverge(net_idx)


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

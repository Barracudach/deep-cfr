
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

from DeepCFR.workers.la.local import LearnerActor as LocalLearnerActor


from .Networks.Actor import Actor 

class TrajectoryBuffer:
    def __init__(self, max_size, state_shape, action_shape):
        self.max_size = max_size
        self.state_shape = state_shape
        self.action_shape = action_shape

        # Инициализация буферов
        self.states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros((max_size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

        self.idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        # Добавление данных в буфер
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        # Обновление индекса и размера
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # Выборка случайного мини-батча
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )
    
@ray.remote(num_cpus=0.1, num_gpus=0.1 if torch.cuda.is_available() else 0)
class TrajectoryWorker():
    pass

class TrajectoryManager():

    def __init__(self,actions):

        GAME_CLASS=DiscretizedNLHoldem
        self.buffer = TrajectoryBuffer(max_size=10000, state_shape=(32,), action_shape=(3,))
        env_args = GAME_CLASS.ARGS_CLS(n_seats=2,
                            bet_sizes_list_as_frac_of_pot=actions,
                            stack_randomization_range=(0, 0,),
                            starting_stack_sizes_list=[50, 50]
                            )
        self.env=GAME_CLASS(env_args,lut_holder=GAME_CLASS.get_lut_holder(),is_evaluating=True)

    def BuildTrajectories(self, agent:Actor, num_episodes=10):

        for episode in range(num_episodes):
            self.env.reset() 
            state=self.env.state_dict()

            episode_reward = 0 

            while True:
                actions_prob = agent.forward(state)

                legal_actions=self.env.get_legal_actions()
                legal_action_probs = actions_prob[legal_actions]
                best_action_index = legal_actions[np.argmax(legal_action_probs)]

                _ , reward, done, __ = self.env.step(best_action_index)
                next_state=self.env.state_dict()
                
                self.buffer.add(state, best_action_index, reward, next_state, done)
             
                state = next_state
                episode_reward += reward

                if done:
                    break

            # Логирование результатов эпизода
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward}")

        # Возвращаем буфер с собранными данными
        return self.buffer
    

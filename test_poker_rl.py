# Copyright (c) 2019 Eric Steinberger


"""
This script will start a game of 3-player No-Limit Texas Hold'em with discrete bet sizes in which you can play against
yourself.
"""

from PokerEnv.InteractiveGame import InteractiveGame
from PokerEnv.games import DiscretizedNLHoldem 

import torch
from MyDeepCfr.Networks import PolicyNetwork, AdvantageNetwork
from MyDeepCfr.Test.Sandbox import *
from MyDeepCfr.EnvWrapper import EnvWrapper

# Загрузка модели
def load_policy_network(model_path, input_size, num_actions):
    model = AdvantageNetwork("pol_net",input_size, num_actions)
    model.load_state_dict(torch.load(model_path)["net"])
    model.eval()  # Переключение в режим оценки
    return model

if __name__ == '__main__':
    
 
    seats_human_plays_list = [1]  # Пользователь играет за место 0
    args = DiscretizedNLHoldem.ARGS_CLS(n_seats=2,
                             bet_sizes_list_as_frac_of_pot=[
                                #  0.2,
                                #  0.5,
                                #  1.0,
                                #  2.0 # Note th1at 1000x pot will always be >pot and thereby represents all-in
                             ],
                             stack_randomization_range=(0, 0,),
                             starting_stack_sizes_list=[40,40],
                             scale_rewards=False
                             )
    env = DiscretizedNLHoldem(env_args=args, is_evaluating=True, lut_holder=DiscretizedNLHoldem.get_lut_holder())
    env_wrapper=EnvWrapper(env)
    env.reset()
    policy_network = load_policy_network("./checkpoints/adv_net0.pt", len(env.get_current_obs(False)["concat"]), env_wrapper.get_actions_count())
    
   
    game = Sandbox(env_wrapper,seats_human_plays_list, policy_network)

    game.start_to_play()
    

# if __name__ == '__main__':
#     game_cls = DiscretizedNLHoldem
#     args = game_cls.ARGS_CLS(n_seats=2,
#                              bet_sizes_list_as_frac_of_pot=[
#                                 0.2,
#                                 0.5,
#                                 1.0,
#                                 2.0
#                              ],
#                              stack_randomization_range=(0, 0,),
#                              starting_stack_sizes_list=[15,15],
#                              scale_rewards=False
#                              )

#     game = InteractiveGame(env_cls=game_cls,
#                            env_args=args,
#                            seats_human_plays_list=[0, 1],
#                            )

#     game.start_to_play()
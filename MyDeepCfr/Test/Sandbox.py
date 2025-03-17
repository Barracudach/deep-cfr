# Copyright (c) 2019 Eric Steinberger
import copy

import numpy as np
from PokerEnv.games import DiscretizedNLHoldem 
from MyDeepCfr.Networks import PolicyNetwork
import torch

class Sandbox:

    def __init__(self, env, seats_human_plays_list, eval_agent:PolicyNetwork=None):

        self._env:DiscretizedNLHoldem=env
        if len(seats_human_plays_list) < env.N_SEATS:
            assert eval_agent is not None

       
        self._env.reset()

        self._eval_agent = eval_agent

        self._seats_human_plays_list = seats_human_plays_list
        self._winnings_per_seat = [0 for _ in range(self._env.N_SEATS)]

    @property
    def seats_human_plays_list(self):
        return copy.deepcopy(self._seats_human_plays_list)

    @property
    def winnings_per_seat(self):
        return copy.deepcopy(self._winnings_per_seat)

    def start_to_play(self, render_mode="TEXT", limit_numpy_digits=True):
        if limit_numpy_digits:
            np.set_printoptions(precision=5, suppress=True)
        print("""                       
                                                _____
                    _____                _____ |6    |
                   |2    | _____        |5    || & & | 
                   |  &  ||3    | _____ | & & || & & | _____
                   |     || & & ||4    ||  &  || & & ||7    |
                   |  &  ||     || & & || & & ||____9|| & & | _____
                   |____Z||  &  ||     ||____S|       |& & &||8    | _____
                          |____E|| & & |              | & & ||& & &||9    |
                                 |____h|              |____L|| & & ||& & &|
                                                             |& & &||& & &|
                                                             |____8||& & &|
                                                                    |____6|
               """)
        self._env.print_tutorial()

        # play forever until human player manually stops
        while True:
            print()
            print("****************************")
            print("*        GAME START        *")
            print("****************************")
            print()

            # ______________________________________________ one episode _______________________________________________

            self._env.reset()
            self._env.render(mode=render_mode)
            while True:
                current_player_id = self._env.current_player.seat_id

                # Human acts
                if current_player_id in self._seats_human_plays_list:
                    strategy = self._env.human_api_ask_action()
                            
                # Agent acts
                else:
                   
                    legal_actions_mask = np.zeros(self._env.N_ACTIONS, dtype="bool")
                    legal_actions_mask[self._env.get_legal_actions()] = 1.0
                  
                    actions_prob = self._eval_agent.get_strategy(
                        torch.tensor(self._env.get_current_obs(False), dtype=torch.float32),
                        torch.tensor(legal_actions_mask, dtype=torch.float32))
                    action_idx=np.argmax(actions_prob)
                    strategy=self._env._get_fixed_action(self._env._get_env_adjusted_action_formulation(action_int=action_idx))
                obs, rews, done, info = self._env._step(strategy)
                self._env.render(mode=render_mode)

                if done:
                    break

            for s in range(self._env.N_SEATS):
                self._winnings_per_seat[s] += np.rint(rews[s] * self._env.REWARD_SCALAR)

            print("")
            print("Current rewards:", rews)
            print("Current Winnings per player:", self._winnings_per_seat)
            input("Press Enter to go to the next round.")

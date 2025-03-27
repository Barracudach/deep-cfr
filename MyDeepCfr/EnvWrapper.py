
import numpy as np
from PokerEnv.games import DiscretizedNLHoldem

class EnvWrapper():


    def __init__(self, env):
        assert issubclass(type(env), DiscretizedNLHoldem)

        self.env:DiscretizedNLHoldem = env
     
    
    def legal_actions(self):
        legal_actions= self.env.get_legal_actions()

        if 3 in legal_actions:#DEBUG
            legal_actions.remove(3)
        return legal_actions
    
    def get_actions_count(self):
        return self.env.N_ACTIONS-1 #DEBUG
    
    def legal_actions_mask(self):
        legal_actions_mask = np.zeros(self.get_actions_count(), dtype="bool")
        legal_actions_mask[self.legal_actions()] = 1.0
        return legal_actions_mask
    

    def _pushback(self, env_obs):
        pass

    def reset(self, deck_state_dict=None):
        return  self.env.reset(deck_state_dict=deck_state_dict)

    def print_obs(self, wrapped_obs):
        assert isinstance(wrapped_obs, np.ndarray)
        print("*****************************************************************************************************")
        print("*****************************************************************************************************")
        print("*****************************************************************************************************")
        print()
        print("________________________________________ OBSERVATION HISTORY ________________________________________")
        print()

        for o in wrapped_obs:
            self.env.print_obs(o)

    def step(self, action):
        self.env.reshuffle_remaining_deck()
        env_obs, rew_for_all_players, done, info = self.env.step(action)
        return env_obs, rew_for_all_players, done, info
    
    def get_current_obs(self):
        return self.env.get_current_obs(False)
    
    def eval_obs(self,stacks:list,pre_betline:list,post_betline:list,board:str,hand:str):
        return self.env.eval_obs(stacks,pre_betline,post_betline,board,hand)
    
    def reshuffle_remaining_deck(self):
        return self.env.reshuffle_remaining_deck()
    
    def state_dict(self):
        return self.env.state_dict()
        # return {
        #     "base": super().state_dict(),
        #     "obs_seq": copy.deepcopy(self._list_of_obs_this_episode)
        # }
    
    def load_state_dict(self, state_dict):
        self.env.load_state_dict(state_dict)



    def seats_count(self):
        return self.env.N_SEATS

    def print_tutorial(self):
        self.env.print_tutorial()

    def render(self, mode='TEXT'):
        self.env.render(mode)

    def get_current_player(self):
        return self.env.current_player
    
    def human_api_ask_action(self):
        return self.env.human_api_ask_action()
    
    def _get_fixed_action(self,action):
        return self.env._get_fixed_action(action)
    
    def get_fixed_action(self,action):
        return self.env._get_fixed_action(action)
    
    def get_reward_scalar(self):
        return self.env.REWARD_SCALAR
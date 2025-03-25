
import numpy as np
from PokerEnv.games import DiscretizedNLHoldem

class EnvWrapper():


    def __init__(self, env):
        assert issubclass(type(env), DiscretizedNLHoldem)

        self.env:DiscretizedNLHoldem = env
        self.action_count=self.env.N_ACTIONS
    
    def legal_actions(self):
        return self.env.get_legal_actions()
    
    def legal_actions_mask(self):
        legal_actions_mask = np.zeros(self.action_count, dtype="bool")
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

   

# Copyright (c) 2019 Eric Steinberger


import copy

import numpy as np

from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs
from PokerRL.game._.wrappers._Wrapper import Wrapper
from PokerRL.game._.rl_env.base.PokerEnv import PokerEnv
from PokerRL.game.games import NoLimitHoldem,DiscretizedNLHoldem

class EnvWrapper():


    def __init__(self, env):
        assert issubclass(type(env), DiscretizedNLHoldem)

        self.env:DiscretizedNLHoldem = env
        self._list_of_obs_this_episode = None
        self.action_count=self.env.N_ACTIONS
    
    def legal_actions(self):
        return self.env.get_legal_actions()
    
    def legal_actions_mask(self):
        legal_actions_mask = np.zeros(self.action_count, dtype="bool")
        legal_actions_mask[self.legal_actions()] = 1.0
        return legal_actions_mask
    
    def _reset_state(self, **kwargs):
        self._list_of_obs_this_episode = []

    def _pushback(self, env_obs):
        pass

    def reset(self, deck_state_dict=None):
        env_obs, rew_for_all_players, done, info = self.env.reset(deck_state_dict=deck_state_dict)
        self._reset_state()

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
        env_obs, rew_for_all_players, done, info = self.env.step(action)
        return env_obs, rew_for_all_players, done, info
    
    def get_current_all_obs(self, env_obs=None):
        return np.array(self._list_of_obs_this_episode, dtype=np.float32)
    
    def get_current_obs(self):
        return self.env.get_current_obs(False)
    
    def state_dict(self):
        return self.env.state_dict()
        # return {
        #     "base": super().state_dict(),
        #     "obs_seq": copy.deepcopy(self._list_of_obs_this_episode)
        # }

    def load_state_dict(self, state_dict):
        self.env.load_state_dict(state_dict)

    def set_to_public_tree_node_state(self, node):
        state_seq = []  # will be sorted. [0] is root.

        def add(_node):
            if _node is not None:
                if _node.p_id_acting_next != _node.tree.CHANCE_ID:
                    self.env.load_state_dict(_node.env_state)
                    state_seq.insert(0, self.env.get_current_obs(is_terminal=False))
                add(_node.parent)

        add(node)  # will step from node to parent to parent of parent... to root.
        self.reset()
        self._reset_state()  # from .reset() the first obs is in by default


        assert len(state_seq) == len(self._list_of_obs_this_episode)

        self.env.load_state_dict(node.env_state, blank_private_info=True)
        assert np.array_equal(node.env_state[EnvDictIdxs.board_2d], self.env.board)

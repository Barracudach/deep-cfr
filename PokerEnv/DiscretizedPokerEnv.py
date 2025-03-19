# Copyright (c) 2019 Eric Steinberger


import time

import numpy as np

from .Poker import Poker
from .PokerEnv import PokerEnv as _PokerEnv
from .poker_env_args import DiscretizedPokerEnvArgs


class DiscretizedPokerEnv(_PokerEnv):
    """
    To discretize No-Limit or Pot-Limit poker games, subclass this baseclass instaed of PokerEnv. It allows to define
    a set of bet_sizes (as fractions of the pot) that are then part of the action space. Contrary to the action format
    of PokerEnv tuple(action, raise_size), discretized envs have integer actions, where 0 is FOLD, 1 is CHECK/CALL and
    then come all specified raise sizes sorted ascending.
    """
    ARGS_CLS = DiscretizedPokerEnvArgs

    def __init__(self,
                 env_args,
                 lut_holder,
                 is_evaluating):

        """
        Args:
            env_args (DiscretePokerEnvArgs):    an instance of DiscretePokerEnvArgs, passing an instance of PokerEnvArgs
                                                will not work.
            is_evaluating (bool):               Whether the environment shall be spawned in evaluation mode
                                                (i.e. no randomization) or not.

        """
        assert isinstance(env_args, DiscretizedPokerEnvArgs)
        assert isinstance(env_args.bet_sizes_list_as_frac_of_pot, list)
        assert isinstance(env_args.uniform_action_interpolation, bool)
        super().__init__(env_args=env_args, lut_holder=lut_holder, is_evaluating=is_evaluating)

        self.bet_sizes_list_as_frac_of_pot = sorted(env_args.bet_sizes_list_as_frac_of_pot)  # ascending
        self.N_ACTIONS = env_args.N_ACTIONS
        self.uniform_action_interpolation = env_args.uniform_action_interpolation

    def _adjust_raise(self, raise_total_amount_in_chips):
        return max(self._get_current_total_min_raise(), raise_total_amount_in_chips)

    def _get_env_adjusted_action_formulation(self, action_int):
        """

        Args:
            action_int: integer representation of discretized action

        Returns:
            list: (action, raise_size) in "continuous" PokerEnv format

        """
        if action_int == 0:
            return [0, -1]
        if action_int == 1:
            return [1, -1]
        if action_int == 2:
            return [2, min(max(self.get_effective_stacks()),self.current_player.stack+self.current_player.current_bet)]
        if action_int == 3:
            return [3, min(self._get_current_total_min_raise(),self.current_player.stack+self.current_player.current_bet)]
        elif action_int >= Poker.BET_RAISE:
            selected = self.get_fraction_of_pot_raise(fraction=self.bet_sizes_list_as_frac_of_pot[action_int - Poker.BET_RAISE],
                                                      player_that_bets=self.current_player)

            if self.uniform_action_interpolation and not self.IS_EVALUATING:
    
                bigger = self.get_fraction_of_pot_raise(fraction=self.bet_sizes_list_as_frac_of_pot[action_int - 1],
                                                        player_that_bets=self.current_player)
                max_amnt = int(float(selected + bigger) / 2)

                # _________________________________________ The minimal amount _________________________________________
                if action_int == Poker.BET_RAISE:  # if lowest bet in repertoire, min is minbet
                    min_amnt = self._get_current_total_min_raise()

                else:  # else, min is the mean of the selected and the next-smaller bet size
                    smaller = self.get_fraction_of_pot_raise(
                        fraction=self.bet_sizes_list_as_frac_of_pot[action_int - 3],
                        player_that_bets=self.current_player)
                    min_amnt = int(float(selected + smaller) / 2)

                if min_amnt >= max_amnt:  # can happen. The sampling would always be the same -> save time.
                    return [Poker.BET_RAISE, min_amnt]
                return [Poker.BET_RAISE, np.random.randint(low=min_amnt, high=max_amnt)]

            else:
                return [Poker.BET_RAISE, selected]
        else:
            raise ValueError(action_int)

    def get_effective_stacks(self):
        if self.N_SEATS==2:
            return [min(self.seats[0].stack+self.seats[0].current_bet,
                       self.seats[1].stack+self.seats[1].current_bet)]*2
        else:
            stacks=[self.seats[i].stack+self.seats[i].current_bet for i in range(self.N_SEATS)]
            sorted_stacks = sorted(stacks)
            n = len(sorted_stacks)
            if n % 2 == 1:
                median_stack = sorted_stacks[n // 2]
            else:
                median_stack = (sorted_stacks[n // 2 - 1] + sorted_stacks[n // 2]) / 2
            max_idx = stacks.index(max(stacks))
            stacks[max_idx] = median_stack    

            return stacks

    def get_legal_actions(self):
        """

        Returns:
            list:   a list with none, one or multiple actions of the set [FOLD, CHECK/CALL, BETSIZE_1, BETSIZE_2, ...]

        """
        legal_actions = []

        # Fold, Check/Call
        for a_int in [Poker.FOLD, Poker.CHECK_CALL]:
            _a = self._get_env_adjusted_action_formulation(action_int=a_int)
            if self._get_fixed_action(action=_a)[0] == a_int:
                legal_actions.append(a_int)

        passed_action=[]
        # since raises are ascending in the list, we can simply loop and break
        for a in range(2, self.N_ACTIONS):  # only loops through raises
            adj_a = self._get_env_adjusted_action_formulation(action_int=a) # берет реально посчитанное действие
            fixed_a = self._get_fixed_action(action=adj_a) #аппроксимация реальной ставки к ближайшей допустимой

            if adj_a[0] != fixed_a[0]:  # if we wanted to raise, but env told us we can not, dont append a raise
                break  
            
            if adj_a[1] < fixed_a[1] or adj_a[1]>self.get_effective_stacks()[self.current_player.seat_id]:
                continue

            if fixed_a[1] in passed_action:
                continue

            passed_action.append(fixed_a[1])
            legal_actions.append(a)  # this action might be modified by env, but is legal and unique -> append

            if adj_a[1] > fixed_a[1]:  # if the raise was too big, an even bigger one will yield the same result
                break
        assert len(legal_actions) > 0
        return legal_actions

    def get_hand_rank(self, hand_2d, board_2d):
        """
        For docs, refer to PokerEnv.get_hand_rank_all_hands_on_given_boards()
        """
        raise NotImplementedError

    def get_hand_rank_all_hands_on_given_boards(self, boards_1d, lut_holder):
        """
        For docs, refer to PokerEnv.get_hand_rank_all_hands_on_given_boards()
        """
        raise NotImplementedError

    def get_random_action(self):
        legal = self.get_legal_actions()
        return legal[np.random.randint(len(legal))]

    def print_tutorial(self):
        print("____________________________________________ TUTORIAL ____________________________________________")
        print("Actions:")
        print("0 \tFold")
        print("1 \tCall")
        print("2 \tAllin")
        print("3 \tMinraise")
        for i in range(4, self.N_ACTIONS):
            print(i, "\tRaise ", self.bet_sizes_list_as_frac_of_pot[i - 4] * 100, "% of the pot")

    def human_api_ask_action(self):
        """ Returns action in Tuple form. """
        while True:
            try:
                print(f'Legal actions: {"".join(map(str, self.get_legal_actions()))}')
                action_idx = int(
                    input("What action do you want to take as player " + str(self.current_player.seat_id) + "?"))
            except ValueError:
                print("You need to enter one of the allowed actions. Refer to the tutorial please.")
                continue
            if (not action_idx in self.get_legal_actions()): # (action_idx < Poker.FOLD or action_idx >= self.N_ACTIONS) or
                print("Invalid action_idx! Please enter one of the allowed actions as described in the tutorial above.")
                continue
            break
        time.sleep(0.01)

        return self._get_fixed_action(self._get_env_adjusted_action_formulation(action_int=action_idx))

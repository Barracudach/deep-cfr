# Copyright (c) 2019 Eric Steinberger


import numpy as np

from .Poker import Poker
from .PokerRange import PokerRange

"""
classes in this file are utilities that poker environments based on PokerEnv can inherit from. They override PokerEnv's
class variables and thereby set it to a certain rule set.
"""

class HoldemRules:
    """
    General rules of Texas Hold'em poker games
    """
    N_HOLE_CARDS = 2
    N_RANKS = 13
    N_SUITS = 4
    N_CARDS_IN_DECK = N_RANKS * N_SUITS
    RANGE_SIZE = PokerRange.get_range_size(n_hole_cards=N_HOLE_CARDS, n_cards_in_deck=N_CARDS_IN_DECK)

    BTN_IS_FIRST_POSTFLOP = False

    N_FLOP_CARDS = 3
    N_TURN_CARDS = 1
    N_RIVER_CARDS = 1
    N_TOTAL_BOARD_CARDS = N_FLOP_CARDS + N_TURN_CARDS + N_RIVER_CARDS
    ALL_ROUNDS_LIST = [Poker.PREFLOP, Poker.FLOP, Poker.TURN, Poker.RIVER]

    SUITS_MATTER = True

    ROUND_BEFORE = {
        Poker.PREFLOP: Poker.PREFLOP,
        Poker.FLOP: Poker.PREFLOP,
        Poker.TURN: Poker.FLOP,
        Poker.RIVER: Poker.TURN
    }
    ROUND_AFTER = {
        Poker.PREFLOP: Poker.FLOP,
        Poker.FLOP: Poker.TURN,
        Poker.TURN: Poker.RIVER,
        Poker.RIVER: None
    }

    RANK_DICT = {
        Poker.CARD_NOT_DEALT_TOKEN_1D: "",
        0: "2",
        1: "3",
        2: "4",
        3: "5",
        4: "6",
        5: "7",
        6: "8",
        7: "9",
        8: "T",
        9: "J",
        10: "Q",
        11: "K",
        12: "A"
    }
    SUIT_DICT = {
        Poker.CARD_NOT_DEALT_TOKEN_1D: "",
        0: "h",
        1: "d",
        2: "s",
        3: "c"
    }
    RANK_DICT_INV =  {v: k for k, v in RANK_DICT.items()}
    SUIT_DICT_INV =  {v: k for k, v in SUIT_DICT.items()}

    STRING = "HOLDEM_RULES"

    def __init__(self):
        from .cpp_wrappers.CppHandeval import CppHandeval

        self._clib = CppHandeval()

    def get_hand_rank_all_hands_on_given_boards(self, boards_1d, lut_holder):
        """
        for docs refer to PokerEnv
        """
        return self._clib.get_hand_rank_all_hands_on_given_boards_52_holdem(boards_1d=boards_1d, lut_holder=lut_holder)

    def get_hand_rank(self, hand_2d, board_2d):
        """
        for docs refer to PokerEnv
        """
        return self._clib.get_hand_rank_52_holdem(hand_2d=hand_2d, board_2d=board_2d)

    @classmethod
    def get_lut_holder(cls):
        from .look_up_table import LutHolderHoldem

        return LutHolderHoldem(cls)


# Copyright (c) 2019 Eric Steinberger


"""
A collection of Poker games often used in computational poker research.
"""

from .Poker import Poker
from .game_rules import HoldemRules
from .DiscretizedPokerEnv import DiscretizedPokerEnv



class DiscretizedNLHoldem(HoldemRules, DiscretizedPokerEnv):
    """
    Discretized version of No-Limit Texas Hold'em (i.e. agents can only select from a predefined set of betsizes)
    """

    RULES = HoldemRules
    IS_FIXED_LIMIT_GAME = False
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 5
    BIG_BLIND = 10
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        HoldemRules.__init__(self)
        DiscretizedPokerEnv.__init__(self,
                                     env_args=env_args,
                                     lut_holder=lut_holder,
                                     is_evaluating=is_evaluating)


"""
register all new envs here!
"""
ALL_ENVS = [
    DiscretizedNLHoldem,
]

from MyDeepCfr.DeepCFRSolver import DeepCFRSolver
from PokerRL.game.games import DiscretizedNLHoldem
from MyDeepCfr.EnvWrapper import EnvWrapper

env_args = DiscretizedNLHoldem.ARGS_CLS(n_seats=3,
                        bet_sizes_list_as_frac_of_pot=[
                            0.2,
                            0.5,
                            1.0,
                            2.0,
                            1000.0
                        ],
                        stack_randomization_range=(0, 0,),
                        starting_stack_sizes_list=[250,250,250]
                        )
env= DiscretizedNLHoldem(env_args,lut_holder=DiscretizedNLHoldem.get_lut_holder(),is_evaluating=True)

env_wrapper=EnvWrapper(env,False)

solver = DeepCFRSolver(
            env_wrapper,
            policy_network_layers=(8, 4),
            advantage_network_layers=(128, 128, 128, 64),
            num_iterations=2,
            num_traversals=2,
            learning_rate=1e-3,
            batch_size_advantage=8,
            batch_size_strategy=8,
            memory_capacity=int(1e7)
        )

solver.solve()




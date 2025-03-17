from MyDeepCfr.GameTreeVisualizer import GameTreeVisualizer
from MyDeepCfr.EnvWrapper import EnvWrapper
from PokerEnv.games import DiscretizedNLHoldem 

args = DiscretizedNLHoldem.ARGS_CLS(n_seats=2,
                            bet_sizes_list_as_frac_of_pot=[
                                0.2,
                                0.5,
                                1.0,
                                2.0 # Note that 1000x pot will always be >pot and thereby represents all-in
                            ],
                            stack_randomization_range=(0, 0,),
                            starting_stack_sizes_list=[15,15],
                            scale_rewards=False
                            )
env = DiscretizedNLHoldem(env_args=args, is_evaluating=True, lut_holder=DiscretizedNLHoldem.get_lut_holder())
env_wrapper=EnvWrapper(env)
env.reset()
    
visualizer = GameTreeVisualizer()
visualizer.reset()
visualizer.traverse_full_tree(env_wrapper.state_dict(), env_wrapper)
visualizer.draw()
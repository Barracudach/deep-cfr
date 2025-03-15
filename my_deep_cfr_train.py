from MyDeepCfr.DeepCFRSolver import DeepCFRSolver
from PokerRL.game.games import DiscretizedNLHoldem
from MyDeepCfr.EnvWrapper import EnvWrapper
import ray

NUM_WORKERS = 4 
NUM_ITERATIONS = 100
SYNT_INTERVAL = 1 


if __name__=="__main__":
    ray.init()

    solver = DeepCFRSolver(
                0,  
                n_seats=3,
                bet_sizes_list_as_frac_of_pot=[
                    0.2,
                    0.5,
                    1.0,
                    2.0,
                    1000.0
                ],
                starting_stack_sizes_list=[1000,1000,1000],
                policy_network_layers=(128, 128, 128, 64),
                advantage_network_layers=(128, 128, 128, 64),
                num_iterations=10,
                num_traversals=5,
                learning_rate=1e-3,
                batch_size_advantage=1000,
                batch_size_strategy=8,
                memory_capacity=int(1e5))


    solver.solve()





from MyDeepCfr.DeepCFRSolver import DeepCFRSolver
from PokerEnv.games import DiscretizedNLHoldem
from MyDeepCfr.EnvWrapper import EnvWrapper
import ray

NUM_WORKERS = 4 
NUM_ITERATIONS = 10000
SYNT_INTERVAL = 1 


if __name__=="__main__":

    solver = DeepCFRSolver(
                0,  
                n_seats=2,
                bet_sizes_list_as_frac_of_pot=[
                #    0.2,
                #     0.5,
                #     1.0,
                #     2.0
                ],
                starting_stack_sizes_list=[40,40],
                num_traversals=250,
                learning_rate=1e-3,
                batch_size_advantage=5000,
                batch_size_strategy=10000,
                memory_capacity=int(1e5),
                scale_rewards=False,
                reinitialize_advantage_networks=True)

    for iteration in range(NUM_ITERATIONS):
        solver.solve()
        if iteration%5==0:
            print("Checkpoint")
            solver.save_all_networks()
            solver.save_memory_state()




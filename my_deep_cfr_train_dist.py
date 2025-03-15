from MyDeepCfr.DeepCFRSolver import  DistDeepCFRSolver
from PokerRL.game.games import DiscretizedNLHoldem
from MyDeepCfr.EnvWrapper import EnvWrapper
import ray

NUM_WORKERS = 4 
NUM_ITERATIONS = 100
SYNT_INTERVAL = 1 


def synchronize_models(solvers:list[DistDeepCFRSolver]):
    all_weights = ray.get([solver.get_model_weights.remote() for solver in solvers])
    avg_weights = {
        "policy_network": average_weights([w["policy_network"] for w in all_weights]),
        "adv_networks": [average_weights([w["adv_networks"][i] for w in all_weights]) for i in range(len(all_weights[0]["adv_networks"]))],
    }

    ray.get([solver.set_model_weights.remote(avg_weights) for solver in solvers])

def average_weights(weights_list):
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = sum(w[key] for w in weights_list) / len(weights_list)
    return avg_weights


if __name__=="__main__":
    ray.init()

    solvers = [DistDeepCFRSolver.remote(
                i,  
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
                memory_capacity=int(1e5)) for i in range(NUM_WORKERS)]



    for iteration in range(NUM_ITERATIONS):
        ray.get([solver.solve.remote() for solver in solvers])

        # Синхронизация моделей каждые sync_interval итераций
        if iteration % SYNT_INTERVAL == 0:
            synchronize_models(solvers)
            print(f"Models synchronized at iteration {iteration}")
        ray.get(solvers[0].save_all_networks.remote())



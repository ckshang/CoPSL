import argparse
import torch
import time
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

from problem_emo import get_problem
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    """
    The lists of the benchmarks:
    F (two-dimensional): ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
    RE (three-dimensional): ['re31', 're32', 're33', 're34', 're37']
    """
    parser.add_argument('--ins_list', type=str, default=
                        ['f1'], help='list of test problems')
    parser.add_argument('--n_run', type=int, default=10, help='number of independent run')
    # EMO
    parser.add_argument('--algorithm', type=str, default='nsga3', help='algorithms: nsga2 / nsga3 / moead')
    parser.add_argument('--n_steps', type=int, default=500, help='number of learning steps')
    parser.add_argument('--n_pref_update', type=int, default=30, help='number of sampled preferences per step')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()

    runtime_list = [0.0] * args.n_run  # record runtime
    hv_list = np.zeros([len(args.ins_list), args.n_steps+1])  # record HV

    # repeatedly run the algorithm n_run times with different seeds
    for run_iter in range(args.n_run):

        seed = args.init_seed + run_iter  # continuous seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        test_problem = 0
        for test_ins in args.ins_list:

            # get problem info
            problem = get_problem(test_ins)
            n_dim = problem.n_dim
            n_obj = problem.n_obj

            # initialize the algorithm
            if args.algorithm == 'nsga2':
                algorithm = NSGA2(pop_size=args.n_pref_update)
            elif args.algorithm == 'nsga3':
                ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=30)
                algorithm = NSGA3(pop_size=args.n_pref_update, ref_dirs=ref_dirs)
            elif args.algorithm == 'moead':
                ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=args.n_pref_update)
                algorithm = MOEAD(ref_dirs, n_neighbors=2, prob_neighbor_mating=0.1)

            # run time
            T1 = time.perf_counter()

            res = minimize(problem, algorithm, ('n_gen', args.n_steps + 1), seed=seed, verbose=False, save_history=True)

            # run time
            T2 = time.perf_counter()
            runtime_list[run_iter] = runtime_list[run_iter] + (T2 - T1)

            obj = {}
            for i in range(args.n_steps + 1):
                generated_pf = []
                generated_ps = []

                obj = problem._evaluate(res.history[i].pop.get('X'), obj)
                generated_ps = res.history[i].pop.get('X')
                generated_pf = obj['F']

                results_F_norm = generated_pf / np.array(problem.nadir_point)

                hv = HV(ref_point=np.array([1.1] * n_obj))
                hv_value = hv(results_F_norm)
                hv_list[test_problem, i] += hv_value

            print('Problem: %s' % test_ins)
            test_problem += 1

        print("************************************************************")

    hv_list = hv_list / args.n_run
    print(np.mean(runtime_list))
    print(hv_list)

    pd.DataFrame(hv_list.T).to_csv('exp_data/EXP_EMO_%s%d.csv' % (args.algorithm, args.init_seed), header=False, index=False)

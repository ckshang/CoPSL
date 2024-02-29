import argparse
import numpy as np
import pandas as pd
import torch
import time
from pymoo.indicators.hv import HV
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.legend_handler import HandlerPathCollection
from matplotlib.ticker import MaxNLocator

from problem import get_problem
from model import PSLModel, CoPSLModel, CoPSLGNModel


def get_args():
    parser = argparse.ArgumentParser()
    """
    The lists of the benchmarks:
    F (two-dimensional): ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
    RE (three-dimensional): ['re31', 're32', 're33', 're34', 're37']
    """
    parser.add_argument('--ins_list', type=str, default=
                        ['f1', 'f2', 'f3', 'f4', 'f5', 'f6'], help='list of test problems')
    parser.add_argument('--n_run', type=int, default=10, help='number of independent run')
    # CoPSL
    parser.add_argument('--loss_func', type=str, default='mtch',
                        help='ls / cosmos / tch / mtch')
    parser.add_argument('--n_steps', type=int, default=500, help='number of learning steps')
    parser.add_argument('--n_pref_update', type=int, default=30, help='number of sampled preferences per step')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=100.0, help='the gamma parameter for cosmos')
    # other settings
    parser.add_argument('--device', type=str, default='cuda', help='the device to run the program')
    parser.add_argument('--init_seed', type=int, default=10, help="random seed")
    args = parser.parse_args()
    return args


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)


class HandlerDashedCollection(HandlerPathCollection):
    def create_collection(self, orig_handle, sizes, *args, **kwargs):
        p = super().create_collection(orig_handle, sizes, *args, **kwargs)
        p.set_edgecolor(p.get_facecolor())  # Set edge color to face color for scatter plot
        return p


def load_pf():
    re31 = pd.read_csv('pf_re/reference_points_RE31.dat', header=None).values
    re32 = pd.read_csv('pf_re/reference_points_RE32.dat', header=None).values
    re33 = pd.read_csv('pf_re/reference_points_RE33.dat', header=None).values
    re34 = pd.read_csv('pf_re/reference_points_RE34.dat', header=None).values
    re37 = pd.read_csv('pf_re/reference_points_RE37.dat', header=None).values

    f1_pf = pd.read_csv('pf_re/reference_points_F1.dat', header=None).values
    re31_pf = []
    re32_pf = []
    re33_pf = []
    re34_pf = []
    re37_pf = []
    for i in range(len(re31)):
        re31_pf_ = list(map(float, re31[i][0].split()))
        re31_pf.append(re31_pf_)
    for i in range(len(re32)):
        re32_pf_ = list(map(float, re32[i][0].split()))
        re32_pf.append(re32_pf_)
    for i in range(len(re33)):
        re33_pf_ = list(map(float, re33[i][0].split()))
        re33_pf.append(re33_pf_)
    for i in range(len(re34)):
        re34_pf_ = list(map(float, re34[i][0].split()))
        re34_pf.append(re34_pf_)
    for i in range(len(re37)):
        re37_pf_ = list(map(float, re37[i][0].split()))
        re37_pf.append(re37_pf_)

    return f1_pf, re31_pf, re32_pf, re33_pf, re34_pf, re37_pf


def evaluate(psmodel, hv_list, t_step):
    psmodel.eval()
    with torch.no_grad():
        generated_pf = []
        generated_ps = []

        if n_obj == 2:
            pref = np.stack([np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)]).T
            pref = pref / np.linalg.norm(pref, axis=1).reshape(len(pref), 1)
            pref = torch.tensor(pref).to(args.device).float()

        if n_obj == 3:
            pref_size = 105
            pref = torch.tensor(das_dennis(13, 3)).to(args.device).float()  # 105

        sol = psmodel(pref)
        for i in range(len(problems)):
            obj = problems[i].evaluate(sol[i])
            generated_ps = sol[i].cpu().numpy()
            generated_pf = obj.cpu().numpy()

            results_F_norm = generated_pf / np.array(problems[i].nadir_point)

            hv = HV(ref_point=np.array([1.1] * n_obj))
            hv_value = hv(results_F_norm)
            hv_list[i, t_step] += hv_value

    return hv_list


if __name__ == '__main__':

    args = get_args()

    runtime_list = [0.0] * args.n_run  # record runtime
    hv_list = np.zeros([len(args.ins_list), args.n_steps + 1])  # record HV
    f1_pf, re31_pf, re32_pf, re33_pf, re34_pf, re37_pf = load_pf()  # load Pareto front

    # repeatedly run the algorithm n_run times with different seeds
    for run_iter in range(args.n_run):

        seed = args.init_seed + run_iter  # continuous seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # get problem info
        problems = [get_problem(problem_i) for problem_i in args.ins_list]
        n_dim = [problems[i].n_dim for i in range(len(problems))]
        n_obj = problems[0].n_obj  # the number of objectives must be the same
        for i in range(len(problems)):
            if problems[i].n_obj != n_obj:
                print("the number of objectives should be consistent")
                exit(1)

        # initialize n_init solutions
        zs = [torch.zeros(n_obj).to(args.device) for i in range(len(problems))]

        # initialize the model and optimizer
        psmodel = CoPSLModel(n_dim, n_obj)
        psmodel.to(args.device)

        # optimizer
        optimizer = torch.optim.Adam(psmodel.parameters(), lr=args.lr)

        # evaluate
        evaluate(psmodel, hv_list, 0)

        # t_step Pareto Set Learning
        for t_step in range(args.n_steps):

            psmodel.train()

            # run time
            T1 = time.perf_counter()

            # sample n_pref_update preferences
            alpha = torch.ones(n_obj, device=args.device)
            pref = torch.distributions.Dirichlet(alpha).sample((args.n_pref_update,))
            pref_vec = pref + 0.0001

            # get the current corresponding solutions
            xs = psmodel(pref_vec)

            # compute losses
            losses = []
            for i, x in enumerate(xs):
                value = problems[i].evaluate(x)

                if args.loss_func == 'ls':
                    ls_value = torch.sum(torch.mul(pref_vec, value), dim=1)
                    losses.append(torch.sum(ls_value))
                elif args.loss_func == 'cosmos':
                    cos = torch.nn.CosineSimilarity(dim=1)
                    cosmos_value = torch.sum(torch.mul(pref_vec, value), dim=1) + torch.mul(args.gamma, cos(pref_vec, value))
                    losses.append(torch.sum(cosmos_value))
                elif args.loss_func == 'tch':
                    tch_value = torch.max(pref_vec * (value - zs[i]), dim=1)[0] + 0.01 * torch.sum(value, dim=1)
                    losses.append(torch.sum(tch_value))
                elif args.loss_func == 'mtch':
                    mtch_value = torch.max((1 / pref_vec) * (value - zs[i]), dim=1)[0] + 0.01 * torch.sum(value, dim=1)
                    losses.append(torch.sum(mtch_value))

            total_loss = torch.sum(torch.stack(losses))  # simply calculate the average of the losses

            # gradient-based pareto set model update
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

            # run time
            T2 = time.perf_counter()
            runtime_list[run_iter] = runtime_list[run_iter] + (T2 - T1)

            # evaluate
            evaluate(psmodel, hv_list, t_step + 1)

        for i in range(len(problems)):
            print('Problem: %s' % args.ins_list[i])

        # config = {
        #     'font.family': 'Times New Roman',
        #     'mathtext.fontset': 'stix'
        # }
        # rcParams['pdf.fonttype'] = 42
        # rcParams['ps.fonttype'] = 42
        # rcParams.update(config)
        # fig = plt.figure(figsize=(8, 6))
        # if n_obj == 3:
        #     selected_problem = 2  # select one of the problems
        #
        #     with torch.no_grad():
        #         pref_size = 105
        #         pref_3d = torch.tensor(das_dennis(50, 3)).to(args.device).float()  # 105
        #         sol = psmodel(pref_3d)
        #         obj = problems[selected_problem].evaluate(sol[selected_problem])
        #         generated_pf = obj.cpu().numpy()
        #         results_F_norm = generated_pf / np.array(problems[selected_problem].nadir_point)
        #         hv = HV(ref_point=np.array([1.1] * n_obj))
        #         hv_value = hv(results_F_norm)
        #
        #     ax = fig.add_subplot(111, projection='3d')
        #
        #     s = 15
        #     if selected_problem == 0:
        #         real_pf = re31_pf
        #     elif selected_problem == 1:
        #         real_pf = re32_pf
        #     elif selected_problem == 2:
        #         real_pf = re33_pf
        #     elif selected_problem == 3:
        #         real_pf = re34_pf
        #     elif selected_problem == 4:
        #         real_pf = re37_pf
        #
        #     ax.scatter(np.array(real_pf)[:, 0], np.array(real_pf)[:, 1], np.array(real_pf)[:, 2], color='#6495ED', s=s, alpha=0.05)
        #     ax.view_init(elev=10., azim=30)
        #     legend_true_pf = ax.scatter([], [], label='Approximated Pareto Front', s=s, alpha=1, color='#6495ED')
        #
        #     ax.scatter(generated_pf[:, 0], generated_pf[:, 1], generated_pf[:, 2], c='#FAA460', s=s, alpha=0.09)
        #     legend_generated_pf = ax.scatter([], [], label='CoPSL Non-Dominated Sols', s=s, alpha=1, color='#FAA460')
        #
        #     ax.legend(handler_map={legend_true_pf: HandlerDashedCollection()}, loc='best', fontsize=20)
        #
        #     plt.tight_layout()
        #
        #     ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='x')
        #     ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
        #     ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='z', useMathText=True)
        #
        #     ax.tick_params(axis='x', labelsize=18, pad=0)
        #     ax.tick_params(axis='y', labelsize=18, pad=0)
        #     ax.tick_params(axis='z', labelsize=18, pad=0)
        #     ax.tick_params(bottom=False, top=False, left=False, right=False)
        #
        #     ax.text(2, 5, 2e9, f'HV: {round(hv_value, 2)}', fontsize=20)
        #
        #     ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
        #
        #     plt.subplots_adjust(right=0.2)
        #
        #     ax.set_xlabel('$f_1$', size=20, labelpad=7)
        #     ax.set_ylabel('$f_2$', size=20, labelpad=8)
        #     ax.set_zlabel('$f_3$', size=20, labelpad=0)
        #
        #     ax.view_init(elev=10., azim=45)
        #
        # elif n_obj == 2:
        #     selected_problem = 0  # select one of the problems
        #
        #     with torch.no_grad():
        #         pref_2d = np.stack([np.linspace(0, 1, 1500), 1 - np.linspace(0, 1, 1500)]).T
        #         pref_2d = pref_2d / np.linalg.norm(pref_2d, axis=1).reshape(len(pref_2d), 1)
        #         pref_2d = torch.tensor(pref_2d).to(args.device).float()
        #         sol = psmodel(pref_2d)
        #         obj = problems[selected_problem].evaluate(sol[selected_problem])
        #         generated_pf = obj.cpu().numpy()
        #         results_F_norm = generated_pf / np.array(problems[selected_problem].nadir_point)
        #         hv = HV(ref_point=np.array([1.1] * n_obj))
        #         hv_value = hv(results_F_norm)
        #
        #     s = 20
        #     plt.scatter(np.array(f1_pf)[:, 0], np.array(f1_pf)[:, 1], s=s, alpha=0.7, color='#6495ED')
        #     legend_true_pf = plt.scatter([], [], label='Approximated Pareto Front', s=s, alpha=1.0, color='#6495ED')
        #     plt.scatter(generated_pf[:, 0], generated_pf[:, 1], s=s, alpha=0.9, color='#FAA460')
        #     legend_generated_pf = plt.scatter([], [], label='CoPSL Non-Dominated Sols', s=s, alpha=1.0, color='#FAA460')
        #     plt.xlabel(r'$f_1$', size=20)
        #     plt.ylabel(r'$f_2$', size=20)
        #     plt.tick_params(bottom=False, top=False, left=False, right=False, labelsize=20)
        #
        #     # Update the legend handler for the True Pareto front
        #     plt.legend(handler_map={legend_true_pf: HandlerDashedCollection()}, loc='best', fontsize=20)
        #     plt.text(x=0.61, y=0.61, s=f'HV: {round(hv_value, 2)}', fontsize=20)
        #
        # plt.grid(linewidth=1)
        # plt.tight_layout()
        # plt.savefig('exp_data/%s_%s_CoPSL_s%d.png' % (args.loss_func, args.ins_list[selected_problem], seed), bbox_inches='tight', dpi=300)
        # # plt.show()

        print("************************************************************")

    hv_list = hv_list / args.n_run
    print(np.mean(runtime_list))
    print(hv_list)

    pd.DataFrame(hv_list.T).to_csv('exp_data/EXP_CoPSL_%s%d.csv' % (args.loss_func, args.init_seed), header=False, index=False)

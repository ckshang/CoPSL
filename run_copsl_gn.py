import argparse
import torch
import time
from pymoo.indicators.hv import HV

from problem import get_problem
from utils import *
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
    # GradNorm
    parser.add_argument('--alpha', type=float, default=1.75, help='alpha in GradNorm')
    # other settings
    parser.add_argument('--device', type=str, default='cuda', help='the device to run the program')
    parser.add_argument('--init_seed', type=int, default=10, help="random seed")
    args = parser.parse_args()
    return args


def evaluate(psmodel, hv_list, t_step):
    psmodel.eval()
    with torch.no_grad():
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

    # repeatedly run the algorithm n_run times
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
        psmodel = CoPSLGNModel(n_dim, n_obj)
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

            total_loss = torch.stack(losses)

            # compute the weighted loss w_i(t) * L_i(t)
            weighted_total_loss = torch.mul(psmodel.weights, total_loss)
            # initialize the initial loss L(0) if t=0
            if t_step == 0:
                if torch.cuda.is_available():
                    initial_total_loss = total_loss.data.cpu()
                else:
                    initial_total_loss = total_loss.data
                initial_total_loss = initial_total_loss.numpy()

            # get the total loss
            loss = torch.sum(weighted_total_loss)

            # gradient-based pareto set model update
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
            psmodel.weights.grad.data = psmodel.weights.grad.data * 0.0

            # get layer of shared weights
            W = psmodel.shared_bottom[-2]  # -1 is the ReLU function

            # get the gradient norms for each of the tasks
            # G^{(i)}_w(t)
            norms = []
            for i in range(len(total_loss)):
                # get the gradient of this task loss with respect to the shared parameters
                gygw = torch.autograd.grad(total_loss[i], W.parameters(), retain_graph=True)
                # compute the norm
                norms.append(torch.norm(torch.mul(psmodel.weights[i], gygw[0])))  # G^{(i)}_{W}(t)
            norms = torch.stack(norms)

            # compute the inverse training rate r_i(t)
            # \curl{L}_i
            if torch.cuda.is_available():
                loss_ratio = total_loss.data.cpu().numpy() / initial_total_loss
            else:
                loss_ratio = total_loss.data.numpy() / initial_total_loss
            # r_i(t)
            inverse_train_rate = loss_ratio / np.mean(loss_ratio)

            # compute the mean norm \tilde{G}_w(t)
            if torch.cuda.is_available():
                mean_norm = np.mean(norms.data.cpu().numpy())
            else:
                mean_norm = np.mean(norms.data.numpy())

            # compute the GradNorm loss
            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=False)
            if torch.cuda.is_available():
                constant_term = constant_term.cuda()
            grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

            # compute the gradient for the weights
            psmodel.weights.grad = torch.autograd.grad(grad_norm_loss, psmodel.weights)[0]

            optimizer.step()

            # renormalize
            normalize_coeff = len(problems) / torch.sum(psmodel.weights.data, dim=0)
            psmodel.weights.data = psmodel.weights.data * normalize_coeff

            # run time
            T2 = time.perf_counter()
            runtime_list[run_iter] = runtime_list[run_iter] + (T2 - T1)

            # evaluate
            evaluate(psmodel, hv_list, t_step + 1)

        for i in range(len(problems)):
            print('Problem: %s' % args.ins_list[i])

        print("************************************************************")

    hv_list = hv_list / args.n_run
    print(np.mean(runtime_list))
    print(hv_list)

    pd.DataFrame(hv_list.T).to_csv('exp_data/EXP_CoPSL_GN_%s%d.csv' % (args.loss_func, args.init_seed), header=False, index=False)


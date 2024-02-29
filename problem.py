import torch
import numpy as np


def get_problem(name, *args, **kwargs):
    name = name.lower()

    PROBLEM = {
        'f1': F1,
        'f2': F2,
        'f3': F3,
        'f4': F4,
        'f5': F5,
        'f6': F6,
        're31': RE31,
        're32': RE32,
        're33': RE33,
        're34': RE34,
        're37': RE37,
    }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](*args, **kwargs)


class F1():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            yi = x[:, i - 1] - torch.pow(2 * x[:, 0] - 1, 2)
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F2():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            theta = 1.0 + 3.0 * (i - 2) / (n - 2)
            yi = x[:, i - 1] - torch.pow(x[:, 0], 0.5 * theta)
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F3():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            xi = x[:, i - 1]
            yi = xi - (torch.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n) + 1) / 2
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0]))

        objs = torch.stack([f1, f2]).T

        return objs


class F4():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - 0.8 * x[:, 0] * torch.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:, 0] * torch.cos(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F5():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - 0.8 * x[:, 0] * torch.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:, 0] * torch.cos((4.0 * np.pi * x[:, 0] + i * np.pi / n) / 3)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F6():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - (0.3 * x[:, 0] ** 2 * torch.cos(12.0 * np.pi * x[:, 0] + 4 * i * np.pi / n) + 0.6 * x[:,
                                                                                                              0]) * torch.sin(
                    6.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - (0.3 * x[:, 0] ** 2 * torch.cos(12.0 * np.pi * x[:, 0] + 4 * i * np.pi / n) + 0.6 * x[:,
                                                                                                              0]) * torch.cos(
                    6.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class RE31():
    def __init__(self, n_dim=3):
        self.problem_name = 'RE31'
        self.n_obj = 3
        self.n_dim = 3
        self.n_constraints = 0
        self.n_original_constraints = 3
        self.nadir_point = [500, 9000000, 20000000]
        self.lbound = torch.tensor([0.00001, 0.00001, 1.0]).float()
        self.ubound = torch.tensor([100, 100, 3.0]).float()

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

        # First original objective function
        f1 = x1 * torch.sqrt(16.0 + (x3 * x3)) + x2 * torch.sqrt(1.0 + x3 * x3)
        # Second original objective function
        f2 = (20.0 * torch.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

        # Constraint functions
        g1 = 0.1 - f1
        g2 = 100000.0 - f2
        g3 = 100000 - ((80.0 * torch.sqrt(1.0 + x3 * x3)) / (x3 * x2))
        g = torch.stack([g1, g2, g3])
        z = torch.zeros(g.shape).cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)
        f3 = torch.sum(g, axis=0).to(torch.float64)
        objs = torch.stack([f1, f2, f3]).T

        return objs


class RE32():
    def __init__(self, n_dim=4):
        self.problem_name = 'RE32'
        self.n_obj = 3
        self.n_dim = 4
        self.n_constraints = 0
        self.n_original_constraints = 4
        self.nadir_point = [35.3096156, 17561.6, 425062977]
        self.ubound = torch.zeros(self.n_dim)
        self.lbound = torch.zeros(self.n_dim)
        self.lbound[0] = 0.125
        self.lbound[1] = 0.1
        self.lbound[2] = 0.1
        self.lbound[3] = 0.125
        self.ubound[0] = 5.0
        self.ubound[1] = 10.0
        self.ubound[2] = 10.0
        self.ubound[3] = 5.0

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        P = 6000
        L = 14
        E = 30 * 1e6

        # // deltaMax = 0.25
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        # First original objective function
        f1 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        # Second original objective function
        f2 = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

        # Constraint functions
        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + torch.pow((x1 + x3) / 2.0, 2)
        R = torch.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + torch.pow((x1 + x3) / 2.0, 2)
        J = 2 * torch.sqrt(torch.tensor(2)) * x1 * x2 * tmpVar

        tauDashDash = (M * R) / J
        tauDash = P / (torch.sqrt(torch.tensor(2)) * x1 * x2)
        tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
        tau = torch.sqrt(tmpVar)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmpVar = 4.013 * E * torch.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
        tmpVar2 = (x3 / (2 * L)) * torch.sqrt(torch.tensor(E / (4 * G)))
        PC = tmpVar * (1 - tmpVar2)

        g1 = tauMax - tau
        g2 = sigmaMax - sigma
        g3 = x4 - x1
        g4 = PC - P

        g = torch.stack([g1, g2, g3, g4])
        z = torch.zeros(g.shape).cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)
        f3 = torch.sum(g, axis=0).to(torch.float64)
        objs = torch.stack([f1, f2, f3]).T

        return objs


class RE33():
    def __init__(self, n_dim=4):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([55, 75, 1000, 11]).float()
        self.ubound = torch.tensor([80, 110, 3000, 20]).float()
        self.nadir_point = [5.3067, 3.12833430979, 25.0]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        # Second original objective function
        f2 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

        # Reformulated objective functions
        g1 = (x2 - x1) - 20.0
        g2 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g3 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / torch.pow((x2 * x2 - x1 * x1), 2)
        g4 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0

        g = torch.stack([g1, g2, g3, g4])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f3 = torch.sum(g, axis=0).float()

        objs = torch.stack([f1, f2, f3]).T

        return objs


class RE34():
    def __init__(self, n_dim=5):
        self.problem_name = 'RE34'
        self.n_obj = 3
        self.n_dim = 5
        self.lbound = torch.tensor([1, 1, 1, 1, 1]).float()
        self.ubound = torch.tensor([3, 3, 3, 3, 3]).float()

        self.nadir_point = [1695.72022, 11.81993945, 0.2903999384]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]

        f1 = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
        f2 = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (0.8364 * x4) - (0.3695 * x1 * x4) + (
                0.0861 * x1 * x5) + (0.3628 * x2 * x4) - (0.1106 * x1 * x1) - (0.3437 * x3 * x3) + (
                     0.1764 * x4 * x4)
        f3 = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (0.0073 * x1 * x2) + (0.024 * x2 * x3) - (
                0.0118 * x2 * x4) - (0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (0.0109 * x4 * x4)

        objs = torch.stack([f1, f2, f3]).T

        return objs


class RE37():
    def __init__(self, n_dim=4):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([0, 0, 0, 0]).float()
        self.ubound = torch.tensor([1, 1, 1, 1]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        xAlpha = x[:, 0]
        xHA = x[:, 1]
        xOA = x[:, 2]
        xOPTT = x[:, 3]

        # f1 (TF_max)
        f1 = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (
                    0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (
                         0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (
                         0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f2 = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (
                    0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (
                         0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (
                         0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f3 = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (
                    0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (
                         0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (
                         0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (
                         0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (
                         0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

        objs = torch.stack([f1, f2, f3]).T

        return objs

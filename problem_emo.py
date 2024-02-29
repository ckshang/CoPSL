import torch
import numpy as np
from pymoo.core.problem import Problem


def get_problem(name, n_dim=10, *args, **kwargs):
    name = name.lower()

    PROBLEM = {
        'f1': F1,
        'f2': F2,
        'f3': F3,
        'f4': F4,
        'f5': F5,
        'f6': F6,
    }
    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](n_dim=n_dim)


class F1(Problem):
    def __init__(self, n_dim=6):
        super().__init__(n_var=6, n_obj=2, n_ieq_constr=0, xl=[0, 0, 0, 0, 0, 0], xu=[1, 1, 1, 1, 1, 1])
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = np.zeros(n_dim)
        self.ubound = np.ones(n_dim)
        self.nadir_point = [1, 1]

    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            yi = x[:, i - 1] - np.power(2 * x[:, 0] - 1, 2)
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - np.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        out['F'] = np.stack([f1, f2]).T

        return out


class F2(Problem):
    def __init__(self, n_dim=6):
        super().__init__(n_var=6, n_obj=2, n_ieq_constr=0, xl=[0, 0, 0, 0, 0, 0], xu=[1, 1, 1, 1, 1, 1])
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = np.zeros(n_dim)
        self.ubound = np.ones(n_dim)
        self.nadir_point = [1, 1]

    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            theta = 1.0 + 3.0 * (i - 2) / (n - 2)
            yi = x[:, i - 1] - np.power(x[:, 0], 0.5 * theta)
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - np.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        out['F'] = np.stack([f1, f2]).T

        return out


class F3(Problem):
    def __init__(self, n_dim=6):
        super().__init__(n_var=6, n_obj=2, n_ieq_constr=0, xl=[0, 0, 0, 0, 0, 0], xu=[1, 1, 1, 1, 1, 1])
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = np.zeros(n_dim)
        self.ubound = np.ones(n_dim)
        self.nadir_point = [1, 1]

    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            xi = x[:, i - 1]
            yi = xi - (np.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n) + 1) / 2
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - np.sqrt(x[:, 0]))

        out['F'] = np.stack([f1, f2]).T

        return out


class F4(Problem):
    def __init__(self, n_dim=6):
        super().__init__(n_var=6, n_obj=2, n_ieq_constr=0, xl=[0, 0, 0, 0, 0, 0], xu=[1, 1, 1, 1, 1, 1])
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = np.zeros(n_dim)
        self.ubound = np.ones(n_dim)
        self.nadir_point = [1, 1]

    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - 0.8 * x[:, 0] * np.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:, 0] * np.cos(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - np.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        out['F'] = np.stack([f1, f2]).T

        return out


class F5(Problem):
    def __init__(self, n_dim=6):
        super().__init__(n_var=6, n_obj=2, n_ieq_constr=0, xl=[0, 0, 0, 0, 0, 0], xu=[1, 1, 1, 1, 1, 1])
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = np.zeros(n_dim)
        self.ubound = np.ones(n_dim)
        self.nadir_point = [1, 1]

    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - 0.8 * x[:, 0] * np.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:, 0] * np.cos((4.0 * np.pi * x[:, 0] + i * np.pi / n) / 3)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - np.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        out['F'] = np.stack([f1, f2]).T

        return out


class F6(Problem):
    def __init__(self, n_dim=6):
        super().__init__(n_var=6, n_obj=2, n_ieq_constr=0, xl=[0, 0, 0, 0, 0, 0], xu=[1, 1, 1, 1, 1, 1])
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = np.zeros(n_dim)
        self.ubound = np.ones(n_dim)
        self.nadir_point = [1, 1]

    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - (0.3 * x[:, 0] ** 2 * np.cos(12.0 * np.pi * x[:, 0] + 4 * i * np.pi / n) + 0.6 * x[:,
                                                                                                              0]) * np.sin(
                    6.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - (0.3 * x[:, 0] ** 2 * np.cos(12.0 * np.pi * x[:, 0] + 4 * i * np.pi / n) + 0.6 * x[:,
                                                                                                              0]) * np.cos(
                    6.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - np.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        out['F'] = np.stack([f1, f2]).T

        return out

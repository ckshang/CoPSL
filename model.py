import torch
import torch.nn as nn


class PSLModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj):
        super(PSLModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj

        self.fc1 = nn.Sequential(
            nn.Linear(self.n_obj, 256),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, self.n_dim),
            nn.Sigmoid(),
        )

    def forward(self, pref):

        x = self.fc1(pref)
        x = self.fc2(x)
        x = self.fc3(x)

        return x.to(torch.float64)


class CoPSLModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj):
        super(CoPSLModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj

        # shared bottom
        self.shared_bottom = nn.Sequential(
            nn.Linear(self.n_obj, 256),
            nn.ReLU(),
        )

        # towers
        for task_i in range(len(self.n_dim)):
            setattr(self, 'task_{}'.format(task_i), nn.Sequential(
                                                        nn.Linear(256, 256),
                                                        nn.ReLU(),
                                                        nn.Linear(256, self.n_dim[task_i]),
                                                        nn.Sigmoid(),
                                                        )
                    )

    def forward(self, pref):

        x = self.shared_bottom(pref)

        outs = []
        for task_i in range(len(self.n_dim)):
            tower = getattr(self, 'task_{}'.format(task_i))
            outs.append(tower(x).to(torch.float64))

        return outs


class CoPSLGNModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj):
        super(CoPSLGNModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj

        # dynamic weights for GradNorm
        self.weights = torch.nn.Parameter(torch.ones(len(self.n_dim)).float())

        # shared bottom
        self.shared_bottom = nn.Sequential(
            nn.Linear(self.n_obj, 256),
            nn.ReLU(),
        )

        # towers
        for task_i in range(len(self.n_dim)):
            setattr(self, 'task_{}'.format(task_i), nn.Sequential(
                                                        nn.Linear(256, 256),
                                                        nn.ReLU(),
                                                        nn.Linear(256, self.n_dim[task_i]),
                                                        nn.Sigmoid(),
                                                        )
                    )

    def forward(self, pref):

        x = self.shared_bottom(pref)

        outs = []
        for task_i in range(len(self.n_dim)):
            tower = getattr(self, 'task_{}'.format(task_i))
            outs.append(tower(x).to(torch.float64))

        return outs

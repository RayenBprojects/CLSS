import torch
from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        num_output_features: int,
        input_features: int = 3,
        use_input_transform: bool = True,
    ):
        super().__init__()

        self.tnet1 = _TNet(input_features) if use_input_transform else None
        self.layers_1 = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.tnet2 = _TNet(64)
        self.layers_2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Linear(128, 1024)
        )

        self.layers_3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_output_features)
        )

    def forward(self, X):
        if self.tnet1 is not None:
            transform = self.tnet1(X)
            X = torch.bmm(X, transform)

        X = self.layers_1(X)

        transform = self.tnet2(X)
        X = torch.bmm(X, transform)

        X = self.layers_2(X)
        X = torch.max(X, dim=1).values

        return self.layers_3(X)


class _TNet(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.input_features= input_features

        self.mlp= nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024)
        )

        self.transform= nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, input_features*input_features),
        )

        nn.init.zeros_(self.transform[-1].weight)
        with torch.no_grad():
            self.transform[-1].bias.copy_(torch.eye(input_features).flatten())

    def forward(self, X):
        batch_size= X.shape[0]

        Y_pred= self.mlp(X)
        Y_pred= torch.max(Y_pred, dim= 1, keepdim= False).values
        return self.transform(Y_pred).reshape(
            batch_size,
            self.input_features,
            self.input_features,
        )
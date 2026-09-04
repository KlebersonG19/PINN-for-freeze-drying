import torch
import torch.nn as nn


class ParametricPINN(nn.Module):
    def __init__(self, t_max=2880.0, L_max=0.05, Pc_max=4.0):
        super(ParametricPINN, self).__init__()
        self.t_max = t_max
        self.L_max = L_max
        self.Pc_max = Pc_max

        # input + first hidden layer
        self.dense1 = nn.Linear(3, 64)
        # next two hidden layers
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, 64)
        # output layer
        self.dense4 = nn.Linear(64, 2)
        
        self.activation = nn.Tanh()
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        t = inputs[:, 0:1] / self.t_max
        L = inputs[:, 1:2] / self.L_max
        Pc = inputs[:, 2:3] / self.Pc_max
        
        x = torch.cat([t, L, Pc], dim=1)
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.activation(self.dense3(x))
        
        out = self.dense4(x)
        
        # Razão de Umidade
        # MR = out[:, 0:1]  ## linear
        MR = torch.relu(out[:, 0:1]) ## com transformação

        # Deff (0 a 30 e-11 m²/s)
        Deff = torch.sigmoid(out[:, 1:2]) * 30.0e-11
        
        return MR, Deff

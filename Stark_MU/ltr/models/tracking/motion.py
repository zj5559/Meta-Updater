import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from ltr import model_constructor
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
class motion_v1(nn.Module):
    def __init__(self,):
        super(motion_v1, self).__init__()
        self.input_proj_m = nn.Linear(4, 64)

        self.m_lstm1 = nn.LSTM(64, 64, 1, batch_first=True)
        self.m_lstm2 = nn.LSTM(64, 64, 1, batch_first=True)
        self.m_lstm3 = nn.LSTM(64, 64, 1, batch_first=True)
        self.fc = nn.Linear(64, 64)
        self.reg=MLP(64, 64, 4, 3)#[x1,y1,x2,y2]
        # self.cls=MLP(256, 256, 1, 2)

        self._initialize_weights()

    def _initialize_weights(self):

        layers = [self.input_proj_m,self.fc,self.reg]
        for layer in layers:
            for name, param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)

    def fbc_motion(self, inputs):
        inputs = self.input_proj_m(inputs)
        outputs, _ = self.m_lstm1(inputs)
        outputs, _ = self.m_lstm2(outputs)
        outputs, _ = self.m_lstm3(outputs)
        return outputs[:,-1]  # [bs,20,64]
    def forward(self,inputs):
        logits_m = self.fbc_motion(inputs)
        logits=self.fc(logits_m)
        offset=self.reg(logits)
        return offset

@model_constructor
def motion_model():
    model = motion_v1()
    device = torch.device('cuda')
    model.to(device)
    return model

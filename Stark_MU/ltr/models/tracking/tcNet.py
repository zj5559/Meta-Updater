import torch.nn as nn
import torch
import torch.nn.functional as F
from ltr.tcopt import tcopts
import numpy as np
from ltr import model_constructor


class tclstm(nn.Module):
    def __init__(self,):
        '''
        in_dim: input feature dim (for bbox is 4)
        hidden_dim: output feature dim
        n_layer: num of hidden layers
        '''
        super(tclstm, self).__init__()
        self.lstm1=nn.LSTM(6,64,1,batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, 1, batch_first=True)
        self.fc=nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 2))
        self._initialize_weights()
    def _initialize_weights(self):
        for layer in self.fc:
            for name,param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
    def net(self, inputs):
        outputs, states=self.lstm1(inputs)
        outputs,states=self.lstm2(outputs[:,-8:])
        outputs, states = self.lstm3(outputs[:,-3:])
        DQN_Inputs = outputs
        outputs=self.fc(outputs[:,-1])
        return outputs, DQN_Inputs
    def forward(self,inputs):
        outputs,DQN_Inputs=self.net(inputs)
        return outputs,DQN_Inputs


class tclstm_fusion(nn.Module):
    #lof_dis取倒数
    def __init__(self,):
        '''
        in_dim: input feature dim (for bbox is 4)
        hidden_dim: output feature dim
        n_layer: num of hidden layers
        '''
        super(tclstm_fusion, self).__init__()
        self.lstm1=nn.LSTM(7,64,1,batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, 1, batch_first=True)
        # self.fc=nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.Linear(64, 2))
        # self.ln = nn.LayerNorm(64)
        self.fc_2=nn.Sequential(
            nn.Linear(64+2, 64),
            nn.Linear(64, 2))
        self._initialize_weights()
    def _initialize_weights(self):
        for layer in self.fc_2:
            for name,param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
    def net(self, inputs):
        outputs, states=self.lstm1(inputs)
        outputs,states=self.lstm2(outputs[:,-8:])
        outputs, states = self.lstm3(outputs[:,-3:])
        # outputs=self.ln(outputs[:,-1])
        # outputs=self.fc(outputs[:,-1])
        return outputs[:,-1], states
    def forward(self,inputs):
        outputs,states=self.net(inputs)
        inputs2 = torch.cat((outputs,inputs[:, -1, -2:]), 1)
        outputs=self.fc_2(inputs2)
        return outputs,states

class SetCriterion(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ce=nn.CrossEntropyLoss()

    def forward(self, output,target):
        loss=self.ce(output,target)
        return loss

@model_constructor
def mu_model():
    model = tclstm_fusion()
    # model = tclstm()
    device = torch.device(tcopts['device'])
    model.to(device)
    return model
def mu_loss():
    criterion = SetCriterion()
    device = torch.device(tcopts['device'])
    criterion.to(device)
    return criterion

'''
model=tclstm()
data=torch.rand(16,20,10)
output,dqn=model.net(data)
print(output.shape)

data=torch.rand(16,1,19,19)
output=model.map_net(data)
print(output.shape)
'''
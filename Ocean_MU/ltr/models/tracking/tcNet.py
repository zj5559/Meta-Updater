import torch.nn as nn
import torch
import torch.nn.functional as F
from ltr.tcopt import tcopts
import numpy as np
from ltr import model_constructor
import time


class tclstm(nn.Module):
    def __init__(self,):
        '''
        in_dim: input feature dim (for bbox is 4)
        hidden_dim: output feature dim
        n_layer: num of hidden layers
        '''
        super(tclstm, self).__init__()
        self.conv_mapNet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, tcopts['map_units'], kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))#mean
        )
        self.lstm1=nn.LSTM(10,64,1,batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, 1, batch_first=True)
        self.fc=nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 2))
        self._initialize_weights()
    def _initialize_weights(self):
        all_layers=[self.conv_mapNet]#,self.fc,self.lstm1,self.lstm2,self.lstm3
        for layer in all_layers:
            for name,param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.normal_(param, 0, 0.01)
        for layer in self.fc:
            for name,param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
    def map_net(self, maps):
        map_outputs=self.conv_mapNet(maps)
        map_outputs=map_outputs.squeeze()#.unsqueeze(1).expand(-1,tcopts['time_steps'],-1)#[bs, tcopts['time_steps'], tcopts['map_units']]
        return map_outputs
    def net(self, inputs):
        outputs, states=self.lstm1(inputs)
        outputs,states=self.lstm2(outputs[:,-8:])
        outputs, states = self.lstm3(outputs[:,-3:])
        DQN_Inputs = outputs
        outputs=self.fc(outputs[:,-1])
        return outputs, DQN_Inputs
    def forward(self,inputs,maps):
        map_outputs=self.map_net(maps)
        map_outputs=map_outputs.reshape(inputs.shape[0],inputs.shape[1],-1)
        inputs=torch.cat((inputs,map_outputs),2)
        outputs,DQN_Inputs=self.net(inputs)
        return outputs,DQN_Inputs,map_outputs
class tclstm_fusion(nn.Module):
    def __init__(self,):
        '''
        in_dim: input feature dim (for bbox is 4)
        hidden_dim: output feature dim
        n_layer: num of hidden layers
        '''
        super(tclstm_fusion, self).__init__()
        self.conv_mapNet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, tcopts['map_units'], kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))#mean
        )
        self.lstm1=nn.LSTM(11,64,1,batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, 1, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, 1, batch_first=True)

        self.fc_2=nn.Sequential(
            nn.Linear(64+2, 64),
            nn.Linear(64, 2))
        self._initialize_weights()
    def _initialize_weights(self):
        all_layers=[self.conv_mapNet]
        for layer in all_layers:
            for name,param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.normal_(param, 0, 0.01)
        for layer in self.fc_2:
            for name,param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
    def map_net(self, maps):
        map_outputs=self.conv_mapNet(maps)
        map_outputs=map_outputs.squeeze()
        return map_outputs
    def net(self, inputs):
        outputs, states=self.lstm1(inputs)
        outputs,states=self.lstm2(outputs[:,-8:])
        outputs, states = self.lstm3(outputs[:,-3:])
        return outputs[:,-1], states
    def forward(self,inputs,maps):
        map_outputs=self.map_net(maps)
        map_outputs=map_outputs.reshape(inputs.shape[0],inputs.shape[1],-1)
        inputs_all=torch.cat((inputs,map_outputs),2)
        outputs,states=self.net(inputs_all)
        inputs2 = torch.cat((outputs,inputs[:, -1, -2:]), 1)
        outputs=self.fc_2(inputs2)
        return outputs,states,map_outputs


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


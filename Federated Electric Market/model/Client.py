import torch
import copy
import numpy as np
from model.models import BiLSTM
from torch import nn
from torch.optim.lr_scheduler import StepLR

from utils.parameter_tran import parameter_to_str, str_to_parameter, get_shape_list
from utils.rsa_algo import rsaEncrypt
from utils.aes_algo import aesDecrypt

# 构建联邦学习的客户端
class client():
    def __init__(self, args, train_data, val_data, test_data, max_load, min_load, rsa_public_k, aes_k):
        self.args = args
        self.model = BiLSTM(args).to(args.device)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.max_load = max_load
        self.min_load = min_load
        self.rsa_public_k = rsa_public_k
        self.aes_k = aes_k

    def update_local_model(self, c):
        m = aesDecrypt(c, self.aes_k)
        shape_list = get_shape_list(self.model)
        params = str_to_parameter(m, shape_list, self.args.round)
        model_params = list(self.model.parameters())
        # 将所有的model_params进行均值处理
        for i in range(len(model_params)):
            model_params[i].data = params[i]

    # def LDP(self, tensor):
    #     tensor_mean = torch.abs(torch.mean(tensor))
    #     tensor = torch.clamp(tensor, min=-self.args.clip, max=self.args.clip)
    #     noise = torch.distributions.laplace.Laplace(0, tensor_mean * self.args.laplace_lambda).sample()
    #     tensor += noise
    #     return tensor

    def train(self):
        print('client is training')
        train_data = self.train_data

        # 定义损失函数MSE
        loss_function = nn.MSELoss().to(self.args.device)
        # 定义优化器
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=self.args.weight_decay)

        scheduler = StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        # 总训练误差
        tol_loss = []

        for epoch in range(self.args.local_epochs):
            train_loss = []
            for (seq, label) in train_data:
                seq, label = seq.to(self.args.device), label.to(self.args.device)
                self.model.zero_grad()
                y_pred = self.model(seq)
                loss = loss_function(y_pred, label)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # 为最后的梯度信息加上Laplace噪声后上传
            tol_loss.append(sum(train_loss)/len(train_loss))
            # 使用gamma调整学习率
            scheduler.step()

            self.model.train()

        # 对所有的模型参数利用rsa公钥进行加密并上传
        model_params = list(self.model.parameters())
        m = parameter_to_str(model_params, self.args.round)
        c = rsaEncrypt(m, self.rsa_public_k, self.args.round)

        return c, sum(tol_loss) / len(tol_loss)
        # return model_list, sum(tol_loss) / len(tol_loss)

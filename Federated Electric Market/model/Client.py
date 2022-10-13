import torch
import copy
import numpy as np
from model.models import BiLSTM
from torch import nn

# 构建联邦学习的客户端
class client():
    def __init__(self, args, train_data, val_data, test_data, max_load, min_load):
        self.args = args
        self.model = BiLSTM(args).to(args.device)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.max_load = max_load
        self.min_load = min_load

    def update_local_model(self, global_model):
        self.model = copy.deepcopy(global_model)

    def LDP(self, tensor):
        tensor_mean = torch.abs(torch.mean(tensor))
        tensor = torch.clamp(tensor, min=-self.args.clip, max=self.args.clip)
        noise = torch.distributions.laplace.Laplace(0, tensor_mean * self.args.laplace_lambda).sample()
        tensor += noise
        return tensor

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

        # 总训练误差
        tol_loss = []
        model_grad = []

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
            for param in list(self.model.parameters()):
                grad = self.LDP(param.grad)
                model_grad.append(grad)
            tol_loss.append(sum(train_loss)/len(train_loss))

        # 返回经过加入拉普拉斯噪声的grad,以及损失
        return model_grad, sum(tol_loss) / len(tol_loss)

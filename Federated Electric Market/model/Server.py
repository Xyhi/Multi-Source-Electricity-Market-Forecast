import torch
import numpy as np
from random import sample
from model.models import BiLSTM
from model_selection.model_train import load_data
from model.Client import client
torch.multiprocessing.set_sharing_strategy('file_system')


# 搭建联邦学习的服务端
class server():
    def __init__(self, args):
        self.args = args
        self.batch_size = args.tol_epochs
        self.lr = args.tol_lr
        self.num_users = args.num_users
        self.frac = args.frac
        self.all_clients = args.all_clients
        self.weight_decay = args.weight_decay
        self.iter = 0   # 返回迭代的次数
        self.client_list = self.build_all_clients()
        self.model = BiLSTM(args).to(args.device)
        self.notice(self.client_list)

    # 构建clients列表
    def build_all_clients(self):
        root_path = 'source_data/data'
        client_list = []
        for i in range(self.num_users):
            train_data, val_data, test_data, max_load, min_load = load_data(self.args, self.args.local_bs,
                                                                            root_path + str(i) + '.xlsx')
            client_list.append(client(self.args, train_data, val_data, test_data, max_load, min_load))
        return client_list

    # 通知所有的client进行模型更新
    def notice(self, clients):
        print('server is distributing current model to clients')
        for one_client in clients:
            one_client.update_local_model(self.model)

    # 将不同的client端传来的parameter_list进行聚合
    def aggregator(self, parameter_list):
        print('server is aggregating......')
        gradient_model = [0] * len(parameter_list[0])
        # 在parameter_list中遍历所有的参数
        for parameter in parameter_list:
            for i in range(len(parameter)):
                gradient_model[i] += parameter[i]

        # gradient这里不是torch类型的变量,所以这里不能直接进行相除,只能逐个相除
        for i in range(len(gradient_model)):
            gradient_model[i] = gradient_model[i] / len(parameter_list)
        # 返回平均的gradient_model,gradient_item,gradient_user
        return gradient_model

    def train(self):
        parameter_list = []
        loss_list = []

        # 如果选择了所有的clients,则对所有的clients进行训练更新
        if self.all_clients:
            clients = self.client_list
        else:
            # 按照一定比例选择client
            m = max(int(self.frac * self.num_users), 1)
            idxs_users = np.random.choice(range(self.num_users), m, replace=False)
            clients = [self.client_list[i] for i in idxs_users]

        for one_client in clients:
            parameter, loss = one_client.train()
            parameter_list.append(parameter)
            loss_list.append(loss)

        gradient_model = self.aggregator(parameter_list)
        model_params = list(self.model.parameters())

        # 将所有的model_params进行均值处理
        for i in range(len(model_params)):
            model_params[i].data = model_params[i].data - self.lr * gradient_model[i] - self.weight_decay * model_params[i].data

        # 通知所有的client进行更新
        self.notice(self.client_list)

        state = {'model': self.model.state_dict()}
        torch.save(state, './network/network{}.pkl'.format(self.iter))

        self.iter += 1
        # 返回损失误差
        return np.mean(loss_list)

import torch
import numpy as np
from model.models import BiLSTM
from model_selection.model_train import load_data
from model.Client import client
from utils.parameter_tran import get_shape_list, str_to_parameter, parameter_to_str
from utils.rsa_algo import rsa_key_generator, rsaDecrypt
from utils.aes_algo import aes_key_generator, aesDecrypt, aesEncrypt
torch.multiprocessing.set_sharing_strategy('file_system')
import os
# 创建文件夹
def build_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)  # 如果不存在目录figure_save_path，则创建

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
        self.round = args.round

        self.iter = 0  # 返回迭代的次数
        self.rsa_public_k, self.rsa_private_k = self.get_rsa_key()  # server端持有rsa的私钥
        self.aes_k = self.get_aes_key()  # client端持有aes密钥
        print('server get key completed')
        self.client_list = self.build_all_clients()
        print('server build_all_clients completed')
        self.model = BiLSTM(args).to(args.device)
        self.notice(self.client_list)

    def get_rsa_key(self):
        pubkey, privkey = rsa_key_generator()
        return pubkey, privkey

    def get_aes_key(self):
        aes_k = aes_key_generator()
        return aes_k

    # 构建clients列表
    def build_all_clients(self):
        root_path = 'source_data/data'
        client_list = []
        for i in range(self.num_users):
            train_data, val_data, test_data, max_load, min_load = load_data(self.args, self.args.local_bs,
                                                                            root_path + str(i) + '.xlsx')
            client_list.append(client(self.args, train_data, val_data, test_data,
                                      max_load, min_load, self.rsa_public_k, self.aes_k))
        return client_list

    # 通知所有的client进行模型更新
    def notice(self, clients, c=None):
        print('server is distributing current model to clients')
        # 第一次将初始化模型给client端
        if c is None:
            model_params = list(self.model.parameters())
            m = bytes(parameter_to_str(model_params, self.args.round), 'utf8')
            init = aesEncrypt(m, self.aes_k)
            for one_client in clients:
                one_client.update_local_model(init)
        else:
            for one_client in clients:
                one_client.update_local_model(c)

    # 将不同的client端传来的parameter_list进行聚合
    def aggregator(self, parameter_list):
        global_model = [0] * len(parameter_list[0])
        for j in range(len(parameter_list[0])):
            for i in range(len(parameter_list)):
                global_model[j] += parameter_list[i][j]
            global_model[j] = torch.div(global_model[j], len(parameter_list))
        return global_model

    # def Fedavg(self, parameter_list):
    #     model_params = list(self.model.parameters())
    #
    #     # 将所有的model_params进行均值处理
    #     for i in range(len(model_params)):
    #         # model_params[i].data = model_params[i].data - self.lr * gradient_model[i] - self.weight_decay * \
    #         #                        model_params[i].data
    #         temp_data = np.zeros_like(parameter_list[i])
    #         for j in range(len(parameter_list)):
    #             temp_data += parameter_list[i][j]
    #         temp_data /= len(parameter_list)
    #
    #         model_params[i].data = torch.tensor(temp_data)

    def train(self,arg_name, arg_val):
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

        for num, one_client in enumerate(clients):
            parameter, loss = one_client.train(num)
            parameter_list.append(parameter)
            loss_list.append(loss)

        parameters = []
        shape_list = get_shape_list(self.model)
        # 对server端传来的parameter进行解密
        print('server is decrypting')
        # for item in parameter_list:
        #     m = rsaDecrypt(item, self.rsa_private_k)
        #     param = str_to_parameter(m, shape_list, self.round)
        #     parameters.append(param)
        # print('server decryption completes')
        # 用aes解密（节省时间）
        for item in parameter_list:
            m = aesDecrypt(item, self.aes_k)
            param = str_to_parameter(m, shape_list, self.round)
            parameters.append(param)
        print('server decryption completes')

        # self.Fedavg(parameter_list)
        gradient_model = self.aggregator(parameters)
        model_params = list(self.model.parameters())
        # 将所有的model_params进行均值处理
        for i in range(len(model_params)):
            model_params[i].data = gradient_model[i]

        model_params = list(self.model.parameters())
        m = bytes(parameter_to_str(model_params, self.args.round), 'utf8')
        c = aesEncrypt(m, self.aes_k)

        # m = parameter_to_str(model_params, self.round)
        # c = rsaEncrypt(m, self.aes_k, self.round)
        # shape_list = get_shape_list(self.model)
        # m = rsaDecrypt(c, self.rsa_private_k)
        # parameters = str_to_parameter(m, shape_list, self.round)
        # c = rsaDecrypt(ret, self.rsa_private_k)
        # 通知所有的client进行更新
        self.notice(self.client_list, c)

        state = {'model': self.model.state_dict()}
        build_dir('./network/{}/{}'.format(arg_name, arg_val))
        torch.save(state, './network/{}/{}/network{}.pkl'.format(arg_name, arg_val,self.iter))

        self.iter += 1
        # 返回损失误差
        return np.mean(loss_list,axis=0)

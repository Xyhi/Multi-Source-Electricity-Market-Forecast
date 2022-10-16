from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from model.Server import server
from utils.options import args
from model_selection.model_train import load_data
from model.Test import test
import warnings
import os
warnings.filterwarnings('ignore')
root_path = 'source_data/data'

# 更新参数：根据输入的要改变的参数和列表，返回一个元素为args的列表
def update_args(args,name,values):
    args_list = []
    for val in values:
        args_temp = deepcopy(args)
        args_temp.__dict__[name] = val
        args_list.append(args_temp)
    return args_list

# 创建文件夹
def build_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)  # 如果不存在目录figure_save_path，则创建

# 保存结果
def save_result(loss_train, tol_test_loss, arg_name, arg_val):
    # 创建文件夹
    build_dir('./result_excel/train_loss/' + arg_name)
    build_dir('./result_excel/test_loss/' + arg_name)
    build_dir('./result_img/train_loss/' + arg_name)
    build_dir('./result_img/test_loss/' + arg_name)

    loss_train = np.array(loss_train)
    tol_test_loss = np.array(tol_test_loss)

    loss_type = ['MSE', 'MAE', 'RMSE']
    # 保存训练集各轮损失值
    loss_train_df = pd.DataFrame(loss_train)
    loss_train_df.columns = loss_type
    loss_train_df.to_excel('./result_excel/train_loss/' + arg_name + '/' + arg_name + '=' + str(arg_val) + '.xlsx')
    # 绘图
    for i in range(3):
        plt.plot(range(len(loss_train)), loss_train[:,i])
        plt.ylabel('train_loss '+loss_type[i])
        plt.savefig(
            './result_img/train_loss/' + arg_name + '/' + arg_name + '=' + str(arg_val) + '-' + loss_type[i] + '.png')
        plt.show()

    # 保存各客户端测试集损失值
    loss_test_df = pd.DataFrame(tol_test_loss)
    loss_test_df.columns = loss_type
    loss_test_df.to_excel('./result_excel/test_loss/' + arg_name + '/' + arg_name + '=' + str(arg_val) + '.xlsx')

    for i in range(3):
        plt.boxplot(tol_test_loss[:,i])
        plt.savefig('./result_img/test_loss/' + arg_name + '/' + arg_name + '=' + str(arg_val) + '-' +loss_type[i] +'.png')
        plt.show()

# 调参过程中的模型训练和测试; arg_name, arg_val用于保存信息
def train_test(server, arg_name, arg_val):
    loss_train = []
    # 进行本地模型训练
    for iter in range(args.tol_epochs):
        local_loss = server.train()
        loss_train.append(local_loss)
        print('ROUND {}: loss(mse,mae,rmse) is {}'.format(iter, local_loss))

    # 测试对所有的client的训练集的误差
    tol_test_loss = []
    final_network = './network/network{}.pkl'.format(args.tol_epochs - 1)
    for idx in range(args.num_users):
        _, _, test_data, max_load, min_load = load_data(args, args.local_bs, root_path + str(idx) + '.xlsx')
        test_loss = test(args, test_data, final_network, max_load, min_load,arg_name, arg_val, idx) # 后三个参数只用于保存信息
        tol_test_loss.append(test_loss)

    print("Average test Loss: ", np.mean(tol_test_loss,axis=0))

    # 保存结果
    save_result(loss_train, tol_test_loss, arg_name, arg_val)


if __name__ == '__main__':

    args = args()



# 调参部分👇
    # 可添加参数对一系列参数进行训练和测试。   update_args: 返回一个元素类型与args()相同的列表
    all_args = {'frac':update_args(args, 'frac', [0.05, 0.1, 0.2, 0.3]),
                }
    # args_list = update_args(args, 'frac', [0.05, 0.1, 0.2, 0.3])
    # print(args_list)
    # 对一系列参数进行训练和测试
# 调参部分👆


    # 训练并测试, 调参时可用for循环遍历args_list
    for name in all_args.keys():
        for i in all_args[name]:
            # 服务端
            Server = server(args)
            # 进入训练和测试
            train_test(Server, name, i.__dict__[name])  # 传入服务端，调参的参数名，对应参数值；后两个参数用于保存信息




    # loss_train = []
    # # 进行本地模型训练
    # for iter in range(args.tol_epochs):
    #     local_loss = server.train()
    #     loss_train.append(local_loss)
    #     print('ROUND {}: loss is {}'.format(iter, local_loss))
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.show()
    #
    # # 测试对所有的client的训练集的误差
    # tol_test_loss = []
    # final_network = './network/network{}.pkl'.format(args.tol_epochs - 1)
    # for idx in range(args.num_users):
    #     _, _, test_data, max_load, min_load = load_data(args, args.local_bs, root_path+str(idx)+'.xlsx')
    #     test_loss = test(args, test_data, final_network, max_load, min_load,idx)
    #     tol_test_loss.append(test_loss)
    #
    # print("Average test Loss: ", np.mean(tol_test_loss))
    # plt.boxplot(tol_test_loss)
    # plt.show()

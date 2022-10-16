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
loss_type = ['MSE', 'MAE', 'RMSE']

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

# 保存训练集误差的表、图；测试集误差的表，图
def save_data(loss_train, tol_test_loss,path1,path2,path3,path4):

    loss_train = np.array(loss_train)
    tol_test_loss = np.array(tol_test_loss)

    # 保存训练集误差
    loss_train_df = pd.DataFrame(loss_train)
    loss_train_df.columns = loss_type
    loss_train_df.to_excel(path1)
    # 绘图
    for i in range(3):
        plt.plot(range(len(loss_train)), loss_train[:,i])
        plt.ylabel('train_loss '+loss_type[i])
        plt.savefig(path2+'-{}.png'.format(loss_type[i]))
        plt.show()

    # 保存测试集误差
    loss_test_df = pd.DataFrame(tol_test_loss)
    loss_test_df.columns = loss_type
    loss_test_df.to_excel(path3)
    # 绘图
    for i in range(3):
        plt.boxplot(tol_test_loss[:,i])
        plt.savefig(path4+'-{}.png'.format(loss_type[i]))
        plt.show()

# 保存结果
def save_result(loss_train, tol_test_loss, arg_name, arg_val):
    # 创建文件夹
    build_dir('./result_excel/train_loss/{}'.format(arg_name))
    build_dir('./result_excel/test_loss/{}'.format(arg_name))
    build_dir('./result_img/train_loss/{}'.format(arg_name))
    build_dir('./result_img/test_loss/{}'.format(arg_name))
    # 训练集误差的表、图；测试集误差的表，图  对应路径
    path1 = './result_excel/train_loss/{}/{}={}.xlsx'.format(arg_name,arg_name,arg_val)
    path2 = './result_img/train_loss/{}/{}={}'.format(arg_name,arg_name,arg_val)
    path3 = './result_excel/test_loss/{}/{}={}.xlsx'.format(arg_name, arg_name, arg_val)
    path4 = './result_img/test_loss/{}/{}={}'.format(arg_name, arg_name, arg_val)
    # 保存
    save_data(loss_train,tol_test_loss,path1,path2,path3,path4)

    # # 保存训练集各轮损失值
    # loss_train_df = pd.DataFrame(loss_train)
    # loss_train_df.columns = loss_type
    # loss_train_df.to_excel('./result_excel/train_loss/{}/{}={}.xlsx'.format(arg_name,arg_name,arg_val))
    # # 绘图
    # for i in range(3):
    #     plt.plot(range(len(loss_train)), loss_train[:,i])
    #     plt.ylabel('train_loss '+loss_type[i])
    #     plt.savefig(
    #         './result_img/train_loss/{}/{}={}-{}.png'.format(arg_name,arg_name,arg_val,loss_type[i]))
    #     plt.show()
    #
    # # 保存各客户端测试集损失值
    # loss_test_df = pd.DataFrame(tol_test_loss)
    # loss_test_df.columns = loss_type
    # loss_test_df.to_excel('./result_excel/test_loss/{}/{}={}.xlsx'.format(arg_name,arg_name,arg_val))
    #
    # for i in range(3):
    #     plt.boxplot(tol_test_loss[:,i])
    #     plt.savefig(
    #         './result_img/test_loss/{}/{}={}-{}.png'.format(arg_name,arg_name,arg_val,loss_type[i]))
    #     plt.show()

# 调参过程中的模型训练和测试; arg_name, arg_val用于保存信息
def train_test(server, arg_name, arg_val):
    loss_train = []
    # 进行本地模型训练
    for iter in range(args.tol_epochs):
        local_loss = server.train(arg_name, arg_val)    # 传入的两个参数用于保存信息
        loss_train.append(local_loss)
        print('ROUND {}: loss(mse,mae,rmse) is {}'.format(iter, local_loss))

    # 测试对所有的client的训练集的误差
    tol_test_loss = []

    final_network = './network/{}/{}/network{}.pkl'.format(arg_name,arg_val,args.tol_epochs - 1)
    for idx in range(args.num_users):
        _, _, test_data, max_load, min_load = load_data(args, args.local_bs, root_path + str(idx) + '.xlsx')
        test_loss = test(args, test_data, final_network, max_load, min_load,arg_name, arg_val, idx) # 后三个参数只用于保存信息
        tol_test_loss.append(test_loss)

    print("Average test Loss: ", np.mean(tol_test_loss,axis=0))

    # 保存结果
    save_result(loss_train, tol_test_loss, arg_name, arg_val)

    return np.mean(loss_train,axis=0), np.mean(tol_test_loss,axis=0)

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


    # 遍历所有参数
    for name in all_args.keys():
        # 创建目录用于保存结果
        build_dir('./result_compare/{}/train'.format(name))
        build_dir('./result_compare/{}/test'.format(name))

        train_loss,test_loss = [], []
        # 遍历参数的所有取值
        for i in all_args[name]:
            print('*******{}={}********'.format(name,i.__dict__[name]))
            # 服务端
            Server = server(i)
            # 进入训练和测试
            train_loss_temp, test_loss_temp = train_test(Server, name, i.__dict__[name])  # 传入服务端，调参的参数名，对应参数值；后两个参数用于保存信息
            train_loss.append(train_loss_temp)
            test_loss.append(test_loss_temp)
        print('参数{}对应误差(mse,mae,rmse):\t训练集：{}\t测试集：{}'.format(name,train_loss,test_loss))
        # 保存同一参数不同结果的对比
        path1 = './result_compare/{}/train/{}.xlsx'.format(name,name)
        path2 = './result_compare/{}/train/{}.png'.format(name, name)
        path3 = './result_compare/{}/test/{}.xlsx'.format(name, name)
        path4 = './result_compare/{}/test/{}.png'.format(name, name)
        save_data(train_loss, test_loss, path1, path2, path3, path4)





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

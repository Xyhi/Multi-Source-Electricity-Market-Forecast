import matplotlib.pyplot as plt
import numpy as np
import torch
from model.Server import server
from utils.options import args
from model_selection.model_train import load_data
from model.Test import test
import warnings
warnings.filterwarnings('ignore')
root_path = 'source_data/data'

if __name__ == '__main__':
    args = args()

    server = server(args)
    loss_train = []
    # 进行本地模型训练
    for iter in range(args.tol_epochs):
        local_loss = server.train()
        loss_train.append(local_loss)
        print('ROUND {}: loss is {}'.format(iter, local_loss))
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.show()

    # 测试对所有的client的训练集的误差
    tol_test_loss = []
    final_network = './network/network{}.pkl'.format(args.tol_epochs - 1)
    for idx in range(args.num_users):
        _, _, test_data, max_load, min_load = load_data(args, args.local_bs, root_path+str(idx)+'.xlsx')
        test_loss = test(args, test_data, final_network, max_load, min_load)
        tol_test_loss.append(test_loss)

    print("Average test Loss: ", np.mean(tol_test_loss))
    plt.boxplot(tol_test_loss)
    plt.show()

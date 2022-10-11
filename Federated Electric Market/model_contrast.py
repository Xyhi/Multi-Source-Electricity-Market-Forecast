import pandas as pd
from utils.options import args
from model_selection.model_train import train, load_data
from model_selection.model_test import test
import matplotlib.pyplot as plt
import numpy as np

# 与没有fedavg算法进行对比

if __name__ == '__main__':
    args = args()

    num_client = args.num_users
    file_list = ['./source_data/data'] * num_client
    name_list = ['block'] * num_client

    tol_loss = []
    for i in range(len(file_list)):
        name_list[i] += str(i)
        file_list[i] += str(i) + '.xlsx'

    for i in range(len(name_list)):
        print('===================client {}==================='.format(i))
        train_data, val_data, test_data, max_load, min_load = load_data(args, args.local_bs, file_list[i])
        loss_list = []
        for j in range(args.local_repeated):
            train(args, train_data, val_data, 'network/network_info.pkl', 1)
            result = test(args, test_data, 'network/network_info.pkl', max_load, min_load, 1)
            loss_list.append(result)
        tol_loss.append(np.mean(loss_list))

    print("Average test Loss: ", np.mean(tol_loss))
    plt.boxplot(tol_loss)
    plt.show()




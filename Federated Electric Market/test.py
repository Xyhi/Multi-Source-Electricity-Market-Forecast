from utils.options import args
from model_selection.model_train import load_data
import matplotlib.pyplot as plt
import numpy as np
from model.Test import test
# 测试对所有的client的训练集的误差
tol_test_loss = []
root_path = 'source_data/data'
args = args()
final_network = './network/network{}.pkl'.format(args.tol_epochs - 1)
for idx in range(args.num_users):
    _, _, test_data, max_load, min_load = load_data(args, args.local_bs, root_path + str(idx) + '.xlsx')
    test_loss = test(args, test_data, final_network, max_load, min_load)
    tol_test_loss.append(test_loss)

print("Average test Loss: ", np.mean(tol_test_loss))
plt.boxplot(tol_test_loss)
plt.show()
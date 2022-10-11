import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
from models.FedAvg import FedAvg
from models.Update import LocalUpdate
from models.models import BiLSTM
from utils.options import args
from model_selection.model_train import load_data
from models.Test import test
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = args()

    # 构建选择的模型
    net_glob = BiLSTM(args).to(args.device)

    # 将网络信息进行存储
    weight_glob = net_glob.state_dict()

    # 模型训练
    loss_train = []
    root_path = './source_data/data'

    # 判断是否选择所有的clients进行训练,如果是则将网络信息复制给所有的clients
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [weight_glob for i in range(args.num_users)]

    # 进行本地模型训练
    for iter in range(args.tol_epochs):
        loss_locals = []
        if not args.all_clients:
            weight_locals = []
        # 按照一定比例选择client
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            train_data, val_data, _, _, _ = load_data(args, args.local_bs, root_path+str(idx)+'.xlsx')
            local = LocalUpdate(args=args, train_data=train_data, val_data=val_data)
            weight, loss = local.loacl_update(copy.deepcopy(net_glob).to(args.device))
            # 将每个client的网络参数进行聚合
            if args.all_clients:
                weight_locals[idx] = copy.deepcopy(weight)
            else:
                weight_locals.append(copy.deepcopy(weight))
            loss_locals.append(copy.deepcopy(loss))
        # server端聚合所有local client的weight
        weight_glob = FedAvg(weight_locals)

        # 将聚合后的网络参数聚合后分发到所有的客户端
        net_glob.load_state_dict(weight_glob)
        # 将每轮结果进行存储
        state = {'models': net_glob.state_dict()}
        torch.save(state, './network/network{}.pkl'.format(iter))
        # 打印所有的平均损失
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average train loss {:.10f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

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

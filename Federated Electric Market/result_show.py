from utils.options import args
from model_selection.model_train import load_data
from model.Test import test
import warnings
warnings.filterwarnings('ignore')
root_path = './source_data/data'

# 测试对所有的client的训练集的误差
if __name__ == '__main__':
    args = args()

    final_network = './network/network{}.pkl'.format(args.tol_epochs - 1)
    for idx in range(0, args.num_users, 10):
        _, _, test_data, max_load, min_load = load_data(args, args.local_bs, root_path + str(idx) + '.xlsx')

        test_loss = test(args, test_data, final_network, max_load, min_load)

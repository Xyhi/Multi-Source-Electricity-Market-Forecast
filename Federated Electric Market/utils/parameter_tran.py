import numpy as np
from copy import deepcopy
import torch


# 将矩阵参数转成字符串
def parameter_to_str(parameter, round):
    m = ''
    for item in parameter:
        # 将其转成numpy进行加密运算
        copy_item = deepcopy(item)
        result = copy_item.data.cpu().numpy()
        for n in np.nditer(result):
            if n > 0:
                n = '%.{}f'.format(round) % n
                m += '+' + n[1:]
            else:
                n = '%.{}f'.format(round) % n
                m += '-' + n[2:]
    return m


# 获取参数的numpy格式
def get_shape_list(model):
    model_params = list(model.parameters())
    shape_list = []
    for i in range(len(model_params)):
        item = deepcopy(model_params[i].data.cpu().numpy())
        if item.ndim == 1:
            shape_list.append((item.shape[0], 1))
        else:
            shape_list.append(item.shape)

    return shape_list


# 根据shape_list返回相应的参数格式
def str_to_parameter(m, shape_list, round):
    start = 0
    end = 0
    block = round + 2
    parameter = []
    for pair in shape_list:
        temp_list = []
        end += pair[0] * pair[1] * block
        sub_m = m[start: end]
        for i in range(int(len(sub_m)/block)):
            temp_list.append(float(sub_m[i*block: (i+1)*block]))

        if pair[1] == 1:
            param = np.array(temp_list).reshape(pair[0])
        else:
            param = np.array(temp_list).reshape(pair)
        parameter.append(torch.tensor(param).float().to('cuda'))
        start = end
    return parameter

import rsa
from utils.parameter_tran import str_to_parameter


# 获取rsa的公钥和私钥
def rsa_key_generator():
    pubkey, privkey = rsa.newkeys(1024)
    return pubkey, privkey


# 利用公钥将m进行加密
def rsaEncrypt(m, pubkey, round):
    # 明文编码格式
    content = m.encode('utf-8')
    block = round + 2
    crypto_list = []
    # 由于考虑到RSA的限制需要对每个block进行分隔
    for i in range(int(len(m) / block)):
        sub_content = content[i * block: (i + 1) * block]
        crypto = rsa.encrypt(sub_content, pubkey)
        crypto_list.append(crypto)

    return crypto_list


# rsa解密
def rsaDecrypt(c, privkey):
    # 所获得的的c应该是个数组
    decrpt_list = ''
    for item in c:
        sub_content = rsa.decrypt(item, privkey)
        decrpt_list += sub_content.decode('utf-8')
    return decrpt_list


if __name__ == '__main__':
    shape_list = [(2, 2), (2, 1)]
    m = '+.11-.12+.13+.14-.15-.16'
    round = 2
    parameter = str_to_parameter(m, shape_list, round)
    print(parameter)
    # a = np.array([0.001, 0.01])
    # for n in np.nditer(a):
    #     n = '%.{}f'.format(3) % n
    #     print(n)

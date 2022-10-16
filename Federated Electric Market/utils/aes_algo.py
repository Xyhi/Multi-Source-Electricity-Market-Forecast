from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import AES

BLOCK_SIZE = 32


# 返回aes的key
def aes_key_generator():
    # 返回一个16byte的强密码
    return 'A92Ui8,!ufala0cM'


# 进行aes加密
def aesEncrypt(m, aes_k):
    cipher = AES.new(aes_k.encode('utf8'), AES.MODE_ECB)
    c = cipher.encrypt(pad(m, BLOCK_SIZE))
    # print(type(c))
    return c


# 进行aes解密
def aesDecrypt(c, aes_k):
    decipher = AES.new(aes_k.encode('utf8'), AES.MODE_ECB)
    temp_m = decipher.decrypt(c)
    m = str(unpad(temp_m, BLOCK_SIZE))[2:-1]
    return m
if __name__ == '__main__':
    aesEncrypt(bytes(8),'A92Ui8,!ufala0cM')
# 这个文件定义了train函数的参数
from utils.VAE import VAE

NUM_TO_LEARN_1 = 0 # 放入训练集的真实图片对数量，数据集共6511张
NUM_TO_LEARN_2 = 4000 # 模拟
EPOCHS = 1000
CUT_EPOCH = 200 # 更换criterion的临界EPOCH数
BATCH_SIZE = 128
LATENTDIM = 256
LR_MAX = 5e-4 # 余弦退火法的开始learn_rate
LR_MIN = 5e-6 # 余弦退火法的结束learn_rate
VAE_INUSE = VAE
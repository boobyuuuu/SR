# 这个文件定义了train函数的参数

# 以下参数均可修改
from utils.VAE import VAE # 选择使用的VAE模型数
# 修改示例：from utils.VAE_EXP_2_3 import VAE
NUM_TO_LEARN_1 = 0 # 放入训练集的真实图片对数量，数据集共6511张
NUM_TO_LEARN_2 = 4000 # 放入训练集的模拟图片对数量，模拟数据集5000+张
EPOCHS = 2000
CUT_EPOCH = 1000 # 更换criterion的临界EPOCH数，一般设置成EPOCH一半
BATCH_SIZE = 128
LATENTDIM = 256 # 这个LATETDIM将导入入VAE
LR_MAX = 5e-4 # 余弦退火法的开始learn_rate
LR_MIN = 5e-6 # 余弦退火法的结束learn_rate

# 以下参数不要改，改VAE在上面
VAE_INUSE = VAE
# 这个文件定义了train函数的参数

NUM_TO_LEARN = 5000 # 放入训练集的图片对数量，数据集共6511张
EPOCHS = 5
CUT_EPOCH = 500 # 更换criterion的临界EPOCH数
BATCH_SIZE = 128
LATENTDIM = 256
LR_MAX = 5e-4 # 余弦退火法的开始learn_rate
LR_MIN = 5e-6 # 余弦退火法的结束learn_rate
MODE = 1 #0代表STED_HC文件训练，1代表使用STED，对应ImageDataset里的 mode 参数。（STED出的模型对泛化能力弱，STED_HC对训练集的还原会有点失真）
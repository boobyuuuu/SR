# train的参数
NUM_TO_LEARN = 5000 #训练集放入图片对数量
EPOCHS = 5
CUT_EPOCH = 500 # 更换criterion的临界EPOCH数
BATCH_SIZE = 128
LATENTDIM = 256
LR_MAX = 5e-4
LR_MIN = 5e-6
MODE = 1 #0代表STED_HC文件训练，1代表使用STED，对应ImageDataset里的 mode 参数。（STED出的模型对泛化能力弱，STED_HC对训练集的还原会有点失真）
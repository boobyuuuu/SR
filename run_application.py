# 这个文件用来对训练出来的模型进行初步判别
from modules.application import application

BATCH_SIZE = 128
EPOCHS = 2000
LATENTDIM = 256
DO_SAVE = 0

application()
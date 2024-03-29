# 这个文件用来启动评估
# 评估的文件再outputs/run_train里
from modules.application import application

BATCH_SIZE = 128
EPOCHS = 2000
LATENTDIM = 256
DO_SAVE = 0

application()
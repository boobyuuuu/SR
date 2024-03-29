# 这个文件定义了一些简单函数
from datetime import datetime
from utils.path_config import folder

def clearlog():
    with open(f'{folder.root()}/run_training.log', 'w') as nothing: # 清空原log
        pass
def log(message):
    with open(f'{folder.root()}/run_training.log', 'a') as f:
        f.write(message + '\n')
        f.write("当前时间：" + str(datetime.now()) + '\n')
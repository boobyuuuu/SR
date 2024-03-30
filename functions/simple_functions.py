# 这个文件定义了一些简单函数
from datetime import datetime
from utils.path_config import folder

# 可以被其他文件调用的函数
def clearlog():
    with open(f'{folder.root()}/train_log.log', 'w') as nothing: # 如果不存在文件就新建，清空原log
        pass
def log(message):
    with open(f'{folder.root()}/train_log.log', 'a') as f:
        f.write(message + '\n')
        f.write("当前时间：" + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '\n')
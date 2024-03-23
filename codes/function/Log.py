# 后台训练时，方便查看训练进度
def log(message):
    with open('training.log', 'a') as f:
        f.write(message + '\n')
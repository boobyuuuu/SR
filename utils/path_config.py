# 这个文件定义了几个目录
root = f'/home/ylin1/SR/VAE256'
class folder:
    @classmethod
    def root(cls):
        return root
    @classmethod
    def Confocal(cls):
        return f'{root}/datasets/Confocal/'
    @classmethod
    def STED(cls):
        return f'{root}/datasets/STED'
    @classmethod
    def STED_HC(cls):
        return f'{root}/datasets/STED_HC/'
    @classmethod
    def run_train(cls):
        return f'{root}/outputs/run_train/'
    @classmethod
    def run_application(cls):
        return f'{root}/outputs/run_application/'


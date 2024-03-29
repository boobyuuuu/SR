# 这个文件指定了几个重要的位置
wd = f'/home/ylin1/SR/VAE256'
class folder:
    @classmethod
    def root(cls):
        return wd
    @classmethod
    def Confocal(cls):
        return f'{wd}/datasets/Confocal/'
    @classmethod
    def STED(cls):
        return f'{wd}/datasets/STED'
    @classmethod
    def STED_HC(cls):
        return f'{wd}/datasets/STED_HC/'
    @classmethod
    def run_train(cls):
        return f'{wd}/outputs/run_train/'
    @classmethod
    def run_application(cls):
        return f'{wd}/outputs/run_application/'


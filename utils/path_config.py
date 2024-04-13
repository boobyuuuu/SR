# 这个文件定义了几个目录
import os
root = os.getcwd()
# 这就是到VAE256这一层
class folder:
    @classmethod
    def root(cls):
        return root
    @classmethod
    def Confocal(cls):
        return f'{root}/datasets/real/Confocal'
    @classmethod
    def STED(cls):
        return f'{root}/datasets/real/STED'
    @classmethod
    def STED_HC(cls):
        return f'{root}/datasets/real/STED_HC'
    @classmethod
    def Confocal_s(cls):
        return f'{root}/datasets/simulated/Confocal'
    @classmethod
    def STED_s(cls):
        return f'{root}/datasets/simulated/STED'
    @classmethod
    def output_train(cls):
        return f'{root}/outputs/train'
    @classmethod
    def output_application(cls):
        return f'{root}/outputs/application'
    @classmethod
    def manual_saves(cls):
        return f'{root}/manual_saves'

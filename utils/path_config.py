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
        return f'{root}/datasets/real/Confocal/'
    @classmethod
    def STED(cls):
        return f'{root}/datasets/real/STED/'
    @classmethod
    def STED_HC(cls):
        return f'{root}/datasets/real/STED_HC/'
    @classmethod
    def Confocal_s(cls):
        return f'{root}/datasets/stimulated/Confocal/'
    @classmethod
    def STED_s(cls):
        return f'{root}/datasets/stimulated/STED/'
    @classmethod
    def output_train(cls):
        return f'{root}/outputs/train_run/'
    @classmethod
    def output_application(cls):
        return f'{root}/outputs/application_run/'
    @classmethod
    def output_demo(cls):
        return f'{root}/outputs/demo_run/'
    @classmethod
    def manual_saves(cls):
        return f'{root}/manual_saves/'

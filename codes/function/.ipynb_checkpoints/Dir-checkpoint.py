# 这个文件用来交代数据集的位置
class Dir:
    @classmethod
    def Confocal(cls):
        return '../datasets/Confocal/'
    @classmethod
    def STED(cls):
        return '../datasets/STED'
    @classmethod
    def STED_HC(cls):
        return '../datasets/STED_HC/'
    @classmethod
    def imgs(cls):
        return '../outputs/applied_imgs'
    @classmethod
    def TEMP(cls):
        return '../outputs/TEMP'
    @classmethod
    def models(cls):
        return '../outputs/trained_models'


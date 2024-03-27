# python 问题与解决

## 1 python 环境同步
每个人都是在自己的账户中操作，每个人都有自己的环境配置。要运行别人的程序，首先进行环境同步

**列出所有环境于requirements.txt：** 

```
pip freeze > requirements.txt
```

**安装requirements.txt的所有环境：**

```
pip install -r requirements.txt
```

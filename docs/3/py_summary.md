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

## 2 jupyter notebook 初次安装后初始配置的问题

在新账户安装完jupyter notebook后，必须先生成和修改配置文件。

1.打开jupyter配置文件，没有就新建

```
vim ~/.jupyter/jupyter_notebook_config.py
```

2.将以下复制进去，也可以搜索并取消原文的注释进行修改。

```
c = get_config()
c.ServerApp.allow_remote_access = True
c.ServerApp.allow_root = True
c.ServerApp.ip = '210.28.140.133' #这个写服务器ip
```

然后点开包含ip的那个链接，能成功连接。

## 3 后台运行ipynb文件

nohup runipy your_notebook.ipynb 2>&1 &

## 4 后台运行py文件

python3 *.py &

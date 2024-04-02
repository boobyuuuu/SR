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

## 2 jupyter notebook 无法连接的问题

运行 `jupyter notebook`后，点开链接，浏览器显示:

```
127.0.0.1 refused to connect.
```

这是因为对jupyter的配置有问题，远程连接失败。解决方法：

1.打开jupyter配置文件

```
vim ~/.jupyter/jupyter_notebook_config.py
```

2.将以下复制进去

```
c = get_config()
c.ServerApp.allow_remote_access = True
c.ServerApp.allow_root = True
c.ServerApp.ip = '210.28.140.133' #这个写服务器ip
```

然后点开包含ip的那个链接，能成功连接。

## 3 后台运行ipynb文件

nohup runipy your_notebook.ipynb 2>&1 &

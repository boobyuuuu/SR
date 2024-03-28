# 服务器问题与解决

此文档用来记录服务器操作过程遇到的 问题 与 解决方法

## 1 权限问题：Permission denied

跨账户操作时常遇到。在一个账户时无法打开另一个账户的文件：

```
ylin1@zhliu-X10DRi:~$ cd /home/zhliu
-bash: cd: /home/zhliu: Permission denied
```

解决方法 1 ：使用sudo 命令

```
sudo + 命令
```

解决方法 2 ：使用root身份操作

```
#进入root管理员身份：
sudo -i

#退出root管理员身份：
exit
```

## 2 查看后台控制台命令

后台运行：

```
nohup runipy ./codes/Train.ipynb 2>&1 &
```

查看后台、查看指定后台、停止后台

注：ps aux 中 Time 的单位是小时（ 7:23 表示 7h23min）

```
# 查看所有后台
ps aux 

# 查看指定用户后台
ps aux | grep username

# 查看 runipy 字符串后台
ps aux | grep runipy

# 停止某个后台
kill PID
```

## 3 账户相关

```
# 查看服务器有哪些账户
getent passwd

# 查看当前账户是否具有root权限
sudo -l

# 查看某用户的所有权限
groups username

# 创建账户
sudo adduser newuser

# 给某用户sudo权限
sudo adduser username sudo
```




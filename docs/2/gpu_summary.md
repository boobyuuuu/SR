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

## 4 系统环境变量 多账户未添加

pip安装相关包时出现：

```
WARNING: The script lit is installed in '/home/zywang/.local/bin' which is not on PATH.
```

并且运行一些命令会找不到。如：`jupyter notebook` ，报错：

```
Jupyter command `jupyter-notebook` not found.
```

解决方法：

1.将当前路径放入系统环境变量中

```
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```
2.应用环境变量

```
source ~/.bashrc
```

## 5 介绍账户隐藏文件

用`ls -a`查看所有文件：

```
.   .bash_history  .bashrc  .config             .ipython  .local    .python_history  .sudo_as_admin_successful
..  .bash_logout   .cache   .ipynb_checkpoints  .jupyter  .profile  snap             .viminfo
```

- . 当前文件
- .. 上一级文件
- .bash_history: 这个文件包含了用户在命令行中执行的历史命令记录。每次用户退出登录时，这个文件会被更新。
- .bash_logout: 当用户退出 Bash shell 时，会执行这个文件中的命令。通常用于清理临时文件或执行其他清理任务。
- **.bashrc: 这是 `Bash shell` 的配置文件，用于设置用户的个性化命令别名、环境变量以及其他 Bash shell 的行为。**
- .cache: 这个目录用于存储应用程序的缓存文件。缓存文件可以提高应用程序的性能，但有时也可能占用大量磁盘空间。
- **.config: 这个目录通常用于存储用户的应用程序配置文件。许多应用程序会在这个目录下创建子目录来存储它们的配置信息。**
- .ipython: 这个目录包含了 IPython（一个交互式 Python shell）的配置文件和历史记录。
- .ipynb_checkpoints: 这个目录是 Jupyter Notebook 自动生成的，用于存储 notebook 文件的检查点版本。这些检查点版本可以用于恢复 notebook 文件的先前状态。
- **.jupyter: 这个目录包含了 Jupyter Notebook 的配置文件和相关数据，例如自定义的笔记本模板和扩展。**
- .local: 这个目录通常用于存储用户的本地安装的程序和数据。例如，用户可以将 Python 包安装到这个目录中，而不是系统范围内安装。
- .profile: 这是用户登录时执行的 Bourne shell 配置文件。它类似于 .bashrc，但适用于 Bourne shell 及其衍生版本，如 Bash。
- snap: 这个目录包含了通过 Snap 包管理器安装的应用程序。Snap 是一种打包和分发 Linux 应用程序的方法，它将应用程序和它们的依赖项捆绑在一起。
- .python_history: 这个文件包含了用户在 Python shell 中执行的历史命令记录，类似于 .bash_history。
- .sudo_as_admin_successful: 这个文件是 sudo 命令生成的，表示上次使用 sudo 命令时身份验证成功。






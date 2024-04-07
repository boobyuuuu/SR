# 超算使用文档

## 相关网址

管理超算账户的主页：
https://scc.nju.edu.cn/

超算2024极简版文档：
https://doc.nju.edu.cn/books/efe93/page/2024-87J

进入超算login账户：（需要两步认证）
https://entry.nju.edu.cn/

收费方法：
https://hpc.nju.edu.cn/zh/hpc/3119-charges

所有计算资源：
https://doc.nju.edu.cn/books/efe93/page/d9640

## 账号 密码

ww_liuzh

8HSu_qL6

## 运行我们的任务需要的操作

1.登陆账号，进入超算login账户

![alt text](1.png)

2.将我们的任务文档传输到一个新文件夹

3.在新文件夹建立运行文档

```
vim job.lsf
```

一个运行文档的示例

```vim linenums="1"
#!/bin/bash
#BSUB -J my_job_name
#BSUB -q my_queue
#BSUB -n 1
#BSUB -W 1:00
#BSUB -gpu num=2
#BSUB -o output.txt
#BSUB -e error.txt

python3 my_script.py
```

4.提交作业

```bash
bsub < job.lsf
```

检查作业

```bash
bjobs
```

删除作业

```bash
bkill id
```




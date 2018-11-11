
# 首先登录nvidia官网。

  https://developer.nvidia.com/cuda-downloads

# 安装cuda 9.0 

  sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  sudo apt-get update
  udo apt-get install cuda-9-0

# 升级cuDNN到7，升级页面如下。

  https://developer.nvidia.com/rdp/cudnn-download

# 下载对应的版本安装。

  sudo dpkg -i libcudnn7_7.4.1.5-1+cuda10.0_amd64.deb
  sudo dpkg -i libcudnn7-dev_7.4.1.5-1+cuda10.0_amd64.deb

#查看cuDNN版本，已经升级到7。
  
    cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
    CUDNN_MAJOR 7
    CUDNN_MINOR 4
    CUDNN_PATCHLEVEL 1

# 推荐使用cuda 9和cuDNN 7，安装最新的tensorflow-gpu版本1.11。
  
    pip install tensorflow-gpu==1.11

# 配置jupyter notebook远程访问

    jupyter notebook --generate-config 

从 jupyter notebook 5.0 版本开始，提供了一个命令来设置密码：jupyter notebook password，生成的密码存储在 jupyter_notebook_config.json。

在 jupyter_notebook_config.py 中找到下面的行，取消注释并修改。

    c.NotebookApp.ip='*'
    c.NotebookApp.password = u'sha:ce...刚才复制的那个密文'
    c.NotebookApp.open_browser = False
    c.NotebookApp.port =8888 #可自行指定一个端口, 访问时使用该端口



# 激活jupyter notebook

    pip install ipykernel
    python -m ipykernel install --user --name tensorflow--display-name "Python (myenv)"

# 代码自动补齐


    pip install jupyter_contrib_nbextensions
    jupyter contrib nbextension install --user
    pip install --user jupyter_nbextensions_configurator
    jupyter nbextensions_configurator enable --user
    

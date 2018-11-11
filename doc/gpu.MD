
首先登录nvidia官网。
https://developer.nvidia.com/cuda-downloads

# 安装cuda 9.0 
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
udo apt-get install cuda-9-0

升级cuDNN到7，升级页面如下。
https://developer.nvidia.com/rdp/cudnn-download
下载对应的版本安装。
sudo dpkg -i libcudnn7_7.4.1.5-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.4.1.5-1+cuda10.0_amd64.deb
查看cuDNN版本，已经升级到7。
cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 4
#define CUDNN_PATCHLEVEL 1

推荐使用cuda 9和cuDNN 7，安装最新的tensorflow-gpu版本1.11。
pip install tensorflow-gpu==1.11

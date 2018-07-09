# 安装advbox
## 安装paddlepaddle
### 创建paddlepaddle环境
通常使用anaconda创建不同的python环境，解决python多版本不兼容的问题。目前advbox仅支持python2.*。

	conda create --name pp python=2.7
	
通过下列命令激活paddlepaddle环境	
	
	source activate pp
	
如果没有安装anaconda，可以通过下载安装脚本并执行。

	wget https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh
	
### 安装paddlepaddle包
最简化的安装可以直接使用pip工具。

	pip install paddlepaddle

如果有特殊需求希望指定版本进行安装，可以使用参数。

	pip install paddlepaddle==0.11.0

如果希望使用GPU加速训练过程，可以安装GPU版本。

	pip install paddlepaddle-gpu

## 安装advbox
advbox以paddlepaddle的models形式出现，可以直接同步paddlepaddle的models代码。

	git clone https://github.com/PaddlePaddle/models.git

在fluid目录下，平级的还有人脸识别、OCR之类的其他模块代码。

	models/fluid# ls
	adversarial   
	face_detection        
	neural_machine_translation
	chinese_ner   
	icnet                 
	object_detection     
	DeepASR       
	image_classification  
	ocr_recognition             

advbox的目录结果如下所示，其中示例代码在tutorials目录下。

	.
	├── advbox
	|   ├── __init__.py
	|   ├── attack
	|        ├── __init__.py
	|        ├── base.py
	|        ├── deepfool.py
	|        ├── gradient_method.py
	|        ├── lbfgs.py
	|        └── saliency.py
	|   ├── models
	|        ├── __init__.py
	|        ├── base.py
	|        └── paddle.py
	|   └── adversary.py
	├── tutorials
	|   ├── __init__.py
	|   ├── mnist_model.py
	|   ├── mnist_tutorial_lbfgs.py
	|   ├── mnist_tutorial_fgsm.py
	|   ├── mnist_tutorial_bim.py
	|   ├── mnist_tutorial_ilcm.py
	|   ├── mnist_tutorial_mifgsm.py
	|   ├── mnist_tutorial_jsma.py
	|   └── mnist_tutorial_deepfool.py
	└── README.md

## hello world
安装完advbox后，可以运行自带的hello world示例代码。
### 生成测试模型
首先需要生成攻击用的模型，advbox的测试模型是一个识别mnist的cnn模型。

	python mnist_model.py

运行完模型后，会将模型的参数保留在当前目录的mnist目录下。查看该目录，可以看到对应的cnn模型的每层的参数，可见有两个卷积层和两个全连接层构成。

	conv2d_0.b_0  
	conv2d_0.w_0  
	conv2d_1.b_0  
	conv2d_1.w_0  
	fc_0.b_0  
	fc_0.w_0  
	fc_1.b_0  
	fc_1.w_0

### 运行攻击代码
这里我们运行下基于FGSM算法的演示代码。

	python mnist_tutorial_fgsm.py

运行攻击脚本，对mnist数据集进行攻击，测试样本数量为500，其中攻击成功394个，占78.8%。

	attack success, original_label=4, adversarial_label=9, count=498
	attack success, original_label=8, adversarial_label=3, count=499
	attack success, original_label=6, adversarial_label=1, count=500
	[TEST_DATASET]: fooling_count=394, total_count=500, fooling_rate=0.788000
	fgsm attack done

# 参考文献

- http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html
- https://github.com/PaddlePaddle/models/tree/develop/fluid/adversarial
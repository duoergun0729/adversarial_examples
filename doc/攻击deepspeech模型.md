# 攻击deepspeech模型
# 概述
N Carlini和D Wagner给出了攻击语音识别模型的方法，攻击的方式属于白盒攻击，攻击的对象是Mozilla实现的百度DeepSpeech模型，该模型已经开源，并且可以下载到训练好的模型参数。

	https://github.com/mozilla/DeepSpeech

他们在论文《Audio Adversarial Examples: Targeted Attacks on Speech-to-Text》中称，无论给出怎样的音频波形，他们都可以制作出另一个与它99.9%相似的音频，而这个音频最终被转换器转换出的文本完全受控制。
# 环境搭建
整个环境需要在python 3.X，比如3.5。
## 安装依赖python库

	pip install  numpy scipy tensorflow pandas python_speech_features pyxdg pydub 

## 同步攻击代码
从github上同步最新的攻击代码。

	git clone https://github.com/carlini/audio_adversarial_examples.git

使用tree命令查看目录结构，其中最重要的就是attack.py，负责生成攻击样本。

	tree .
	.
	├── LICENSE
	├── README
	├── attack.py
	├── classify.py
	├── filterbanks.npy
	├── make_checkpoint.py
	└── tf_logits.py
	
	0 directories, 7 files

attack.py攻击脚本的调用方式为，

	usage: attack.py [-h] --in INPUT [INPUT ...] --target TARGET
	                 [--out OUT [OUT ...]] [--outprefix OUTPREFIX]
	                 [--finetune FINETUNE [FINETUNE ...]] [--lr LR]
	                 [--iterations ITERATIONS] [--l2penalty L2PENALTY] [--mp3]

其中最主要的参数含义如下：

- in，被攻击的wav文件
- target，伪装的内容
- out，生成的对抗样本文件名

以攻击代码目录为当前目录完成剩下的安装步骤。

## 同步DeepSpeech
从github上同步最新的DeepSpeech代码，同步的过程中会下载最新的模型参数。

	git clone https://github.com/mozilla/DeepSpeech.git

安装DeepSpeech工具。
	
	pip install DeepSpeech

安装了DeepSpeech工具后就可以使用对应的命令行进行语音识别了。

	usage: deepspeech [-h] model audio alphabet [lm] [trie]

其中最重要参数含义如下：

	- model，模型文件
	- audio，需要识别的wav文件
	- alphabet，映射文件

## 下载指定版本模型参数文件
语音识别模型需要在大量的样本上训练才能获得可用的模型参数。下载指定版本模型参数用于生成攻击样本，本例中为0.1.0版。

	wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
	tar -xzf deepspeech-0.1.0-models.tar.gz

在攻击代码的目录下解压文件，其中output_graph.pb和lm.binary是模型参数文件，alphabet.txt是映射表。

	tree .
	.
	├── alphabet.txt
	├── lm.binary
	├── output_graph.pb
	└── trie
	
	0 directories, 4 files

## 生成checkpoint文件
下载了下载指定版本模型参数文件后，需要针对该文件生成对应的checkpoint文件。在make_checkpoint.py中指定对应的模型参数文件的名称。

	#loaded = graph_def.ParseFromString(open("models/saved_model.pb","rb").read())
	loaded = graph_def.ParseFromString(open("models/output_graph.pb","rb").read())

在make_checkpoint.py中指定被攻击的wav文件的名称。

    #mfcc = audiofile_to_input_vector("sample.wav", 26, 9)
    mfcc = audiofile_to_input_vector("case1.wav", 26, 9)

运行make_checkpoint.py，生成checkpoint文件。


至此已经完成了完整的配置过程，以攻击代码目录为当前目录，查看文件，其中DeepSpeech文件夹为DeepSpeech代码目录，models为解压deepspeech-0.1.0-models.tar.gz文件后生成的目录。

	tree -L 1 .
	.
	├── DeepSpeech
	├── LICENSE
	├── README
	├── attack.py
	├── classify.py
	├── filterbanks.npy
	├── make_checkpoint.py
	├── models
	└── tf_logits.py

# 样本数据
测试样本可以直接使用Common Voice数据集，下载方式如下所示。Common Voice数据集是世界上种类最多的公开语音数据集，便于全世界的研究人员开发和优化的语音训练技术。

	wget https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz

也可以通过百度、讯飞等提供的免费服务生成指定的语音样本，通常生成的语音样本都是mp3格式，但是在本例中仅指出wav格式，所以需要进行格式转换。mac下提供了命令行工具ffmpeg，非常方便的进行装换，其安装方式如下。

	brew install ffmpeg  

ffmpeg转换mp3文件到wav文件的方式为：
	
	ffmpeg -i mp3文件  生成wav文件名

假设case1.mp3的内容为经典格言：
	
	"It was the best of times, it was the worst of times"

使用ffmpeg把case1.mp3转换为wav格式。

	ffmpeg -i case1.mp3 case1.wav

显示内容如下所示，可见默认是16K Hz，编码方式为pcm。

	Input #0, mp3, from 'case1.mp3':
	  Duration: 00:00:04.36, start: 0.000000, bitrate: 16 kb/s
	    Stream #0:0: Audio: mp3, 16000 Hz, mono, fltp, 16 kb/s
	Stream mapping:
	  Stream #0:0 -> #0:0 (mp3 (mp3float) -> pcm_s16le (native))
	Press [q] to stop, [?] for help
	Output #0, wav, to 'case1.wav':
	  Metadata:
	    ISFT            : Lavf58.12.100
	    Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s
	    Metadata:
	      encoder         : Lavc58.18.100 pcm_s16le
	size=     136kB time=00:00:04.35 bitrate= 256.1kbits/s speed= 603x    
	video:0kB audio:136kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.055957%

使用DeepSpeech命令行工具进行语音识别。

	deepspeech ../models/output_graph.pb case1.wav ../models/alphabet.txt 

结果基本满足需求，使用的模型是0.1.0版本的。

	Loading model from file ../models/output_graph.pb
	2018-05-28 17:21:11.667978: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
	Loaded model in 3.348s.
	Running inference.
	iti was the best of tines it it was the worst of time
	Inference took 28.761s for 4.356s audio file.

# 生成攻击样本
使用攻击脚本生成攻击样本，伪装的内容如下：
	
	"hello world hello deeplearning"

执行攻击脚本，被攻击文件为case1.wav。

	python attack.py --in case1.wav  --target "hello world hello deeplearning" --out adversarial01.wav

使用DeepSpeech命令行工具进行语音识别。

	deepspeech models/output_graph.pb adversarial01.wav models/alphabet.txt

识别内容满足预期，为"hello world hello deeplearning"，攻击成功。

	Loading model from file models/output_graph.pb
	2018-05-29 11:14:41.300083: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
	2018-05-29 11:14:41.439620: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
	2018-05-29 11:14:41.440021: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
	name: Tesla P4 major: 6 minor: 1 memoryClockRate(GHz): 1.1135
	pciBusID: 0000:00:06.0
	totalMemory: 7.43GiB freeMemory: 7.32GiB
	2018-05-29 11:14:41.440057: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: Tesla P4, pci bus id: 0000:00:06.0, compute capability: 6.1)
	Loaded model in 0.534s.
	Running inference.
	hello world hello deeplearning
	Inference took 3.669s for 4.356s audio file.

# 性能优化
为了提升性能，可以使用GPU加速，tensorflow和DeepSpeech可以使用GPU的版本。

	pip install DeepSpeech-gpu
	pip install tensorflow-gpu

# 参考文献
- https://github.com/carlini/audio_adversarial_examples
- https://github.com/mozilla/DeepSpeech
- N Carlini，D Wagner，Audio Adversarial Examples: Targeted Attacks on Speech-to-Text，arXiv:1801.01944
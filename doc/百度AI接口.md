# 语音合成
我们介绍最常用的Python SDK。Python SDK需要联网调用http接口，支持最多512字（1024 字节)的音频合成，合成的文件格式为mp3。

## 申请API Key
使用语音合成服务之前，需要在百度云的控制台申请服务密钥，填完申请信息后会告知属于你的AppID、API Key和Secret Key。

## 安装语音合成 Python SDK
从网站上下载最新的语音合成 Python SDK安装包。

	https://ai.baidu.com/sdk#asr

语音合成Python SDK目录结构如下所示：

	├── README.md
	├── aip                   //SDK目录
	│   ├── __init__.py       //导出类
	│   ├── base.py           //aip基类
	│   ├── http.py           //http请求
	│   └── speech.py //语音合成
	└── setup.py              //setuptools安装
	
也可以直接使用pip进行安装。

	pip install baidu-aip

## 新建AipSpeech
AipSpeech是语音识别的Python SDK客户端，为使用语音识别的开发人员提供了一系列的交互方法。

	from aip import AipSpeech
	
	""" 你的 APPID AK SK """
	APP_ID = '你的 App ID'
	API_KEY = '你的 Api Key'
	SECRET_KEY = '你的 Secret Key'
	
	client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

## 合成语音
合成语音的接口非常简便。

	client.synthesis(tex,lang,ctp,opt)

其中比较重要的几个参数含义如下：

- tex，合成的文本
- lang，语言选择,填写zh，即使生成英文也要设置为zh
- ctp，客户端类型选择，web端填写1

可选的几个参数含义如下：

- cuid，用户唯一标识填
- spd, 语速，取值0-9，默认为5中语速
- pit，音调，取值0-9，默认为5中语调
- vol，音量，取值0-15，默认为5中音量

典型的错误码如下：

- 500，不支持的输入
- 501，输入参数不正确
- 502，token验证失败
- 503，合成后端错误

简单封装后就可以直接使用。

	def  text2mp3(text,filename):
	    result = client.synthesis(text, 'zh', 1, {
	        'vol': 5,
	    })
		
	    # 识别正确返回语音二进制 错误则返回dict 
	    if not isinstance(result, dict):
	        with open(filename, 'wb') as f:
	            f.write(result)
	    else:
	        print dict

## mp3转换成wav
语音合成只支持生成mp3格式的文件，但是部分场合需要使用wav格式的文件。一种方法是通过ffmpeg命令直接转换。

	ffmpeg -i love.mp3   love.wav

运行结果如下。

	Input #0, mp3, from 'hello.mp3':
	  Duration: 00:00:01.08, start: 0.000000, bitrate: 16 kb/s
	    Stream #0:0: Audio: mp3, 16000 Hz, mono, fltp, 16 kb/s
	Stream mapping:
	  Stream #0:0 -> #0:0 (mp3 (mp3float) -> pcm_s16le (native))
	Press [q] to stop, [?] for help
	Output #0, wav, to 'hello.wav':
	  Metadata:
	    ISFT            : Lavf58.12.100
	    Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s
	    Metadata:
	      encoder         : Lavc58.18.100 pcm_s16le
	size=      34kB time=00:00:01.08 bitrate= 256.6kbits/s speed= 139x    
	video:0kB audio:34kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.225694%


mac下ffmpeg的安装非常简便。

	brew install ffmpeg


# 参考文献
- http://ai.baidu.com/docs
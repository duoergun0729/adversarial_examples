# -*- coding: UTF-8 -*-
from aip import AipSpeech


"""
语音合成
免费使用
http://tsn.baidu.com/text2audio
200000次/天免费
不保证并发
"""


APP_ID = '11300174'
API_KEY = 'TsZiyUPjGchEQUhfZzvB6LGh'
SECRET_KEY = 'mPSjW1L4Elc3VVa2CXyLDVlK2nVQfqWp'


client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


#语音合成
def  text2mp3(text,filename):
    result = client.synthesis(text, 'zh', 1, {
        'vol': 5,
    })

    # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
    if not isinstance(result, dict):
        with open(filename, 'wb') as f:
            f.write(result)
    else:
        print(dict)

# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()




if __name__ == '__main__':

    #text2mp3("It was the best of times, it was the worst of times","case1.mp3")

    # 识别本地文件
    result=client.asr(get_file_content('adversarial01.wav'), 'wav', 16000, {
        'dev_pid': 1737,
    })



    print(result)


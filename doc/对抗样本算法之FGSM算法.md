# 对抗样本算法之FGSM算法
## 概述
在前面文章《对抗样本的基本原理》中，我们介绍了生成对抗样本的基本思路，其中大体思路分为白盒攻击和黑盒攻击，区别在于黑盒测试把模型当做黑盒，只能输入样本获得预测结果，白盒在黑盒的基础上还可以获取模型的参数、梯度等信息。本文将介绍白盒攻击中鼎鼎大名的FGSM（Fast Gradient Sign Method）算法。

# 参考文献
- Ian J. Goodfellow, Jonathon Shlens & Christian Szegedy，Explaining and Harnessing Adversarial Examples，arXiv:1412.6572
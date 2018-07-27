# -*- coding: utf-8 -*-

"""
FGSM tutorial on cifar10 using advbox tool.
FGSM method is non-targeted attack while FGSMT is targeted attack.
"""

#docker run --name paddle-test -v $PWD:/paddle --network=host -it paddlepaddle/paddle /bin/bash


import sys
sys.path.append("..")


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np
import paddle.fluid as fluid
import paddle.v2 as paddle

from advbox.adversary import Adversary
#from advbox.attacks.gradient_method import FGSM
from advbox.attacks.gradient_method import FGSMT
from advbox.models.paddle import PaddleModel
from tutorials.mnist_model import mnist_cnn_model
#from vgg import vgg_bn_drop
#from resnet import resnet_cifar10

from PIL import Image

def main():
    """
    Advbox demo which demonstrate how to use advbox.
    """
    TOTAL_NUM = 50
    IMG_NAME = 'img'
    LABEL_NAME = 'label'

    img = fluid.layers.data(name=IMG_NAME, shape=[1, 28, 28], dtype='float32')
    # gradient should flow
    img.stop_gradient = False
    label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')
    
    logits = mnist_cnn_model(img)
    #logits = vgg_bn_drop(img)
    #logits = resnet_cifar10(img,32)
    
    cost = fluid.layers.cross_entropy(input=logits, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # use CPU
    place = fluid.CPUPlace()
    # use GPU
    #place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    BATCH_SIZE = 1
    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.test(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    fluid.io.load_params(
        exe, "mnist/", main_program=fluid.default_main_program())

    # advbox demo
    m = PaddleModel(
        fluid.default_main_program(),
        IMG_NAME,
        LABEL_NAME,
        logits.name,
        avg_cost.name, (-1, 1),
        channel_axis=1)
    #attack = FGSM(m)
    attack = FGSMT(m)
    attack_config = {"epsilons": 0.3}
    # use test data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    for data in test_reader():
        total_count += 1
        adversary = Adversary(data[0][0], data[0][1])

        # FGSM non-targeted attack
        #adversary = attack(adversary, **attack_config)

        # FGSMT targeted attack
        tlabel = 8
        adversary.set_target(is_targeted_attack=True, target_label=tlabel)
        adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (data[0][1], adversary.adversarial_label, total_count))

            adversarial_example=adversary.adversarial_example
            #print adversarial_example

            #原始数据归一化到(-1,1)之间了 需要还原到(0,255)

            adversarial_example /= 2.
            adversarial_example += 0.5
            adversarial_example *= 255.

            #im = Image.fromarray(adversarial_example.astype(np.uint8))

            #filename= "%d-%d.jpg" % (adversary.original_label,adversary.target_label)

            #im.save("output/"+filename)

        else:
            print('attack failed, original_label=%d, count=%d' %
                  (data[0][1], total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break
    print("fgsmt attack done")


if __name__ == '__main__':
    main()

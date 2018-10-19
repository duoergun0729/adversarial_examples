# -- coding: utf-8 --
"""
DeepFool tutorial on mnist using advbox tool.
Deepfool is a simple and accurate adversarial attack method.
It supports both targeted attack and non-targeted attack.
"""
import sys
import pickle
sys.path.append("..")

#import matplotlib.pyplot as plt
import paddle.fluid as fluid
import paddle.v2 as paddle

from advbox.adversary import Adversary
from advbox.attacks.deepfool import DeepFoolAttack
from advbox.models.paddle import PaddleModel
#from tutorials.mnist_model import mnist_cnn_model
#from mnist_model_blackbox import mnist_mlp_model

modela_path="./mlp_harden/mlp/"
modelb_path="./mlp_harden/mlp-plus/"

adversarial_examples_pkl="adversarial_examples.pkl"

# 1表示使用GPU
use_cuda = 1


# 全局配置
BATCH_SIZE = 128
PASS_NUM = 10


#生成对抗样本个数
TOTAL_NUM = 50


def mnist_mlp_model(img):

    fc1 = fluid.layers.fc(input=img, size=512, act='relu')

    fc2 = fluid.layers.fc(input=fc1, size=512, act='relu')

    logits = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return logits



def make_mlp_model_new(dirname):



    """
    Train the cnn model on mnist datasets
    """
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    logits = mnist_mlp_model(img)
    cost = fluid.layers.cross_entropy(input=logits, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    optimizer = fluid.optimizer.Adam(learning_rate=0.01)
    optimizer.minimize(avg_cost)

    # 每个batch计算的时候能取到当前batch里面样本的个数，从而来求平均的准确率
    batch_size = fluid.layers.create_tensor(dtype='int64')

    batch_acc = fluid.layers.accuracy(
        input=logits, label=label, total=batch_size)

    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

    # set to True if training with GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(fluid.default_startup_program())

    # 测试程序
    inference_program = fluid.default_main_program().clone(for_test=True)

    accuracy = fluid.average.WeightedAverage()
    test_accuracy = fluid.average.WeightedAverage()
    # 开始训练，使用循环的方式来指定训多少个Pass
    for pass_id in range(PASS_NUM):
        # 从训练数据中按照一个个batch来读取数据
        accuracy.reset()
        for batch_id, data in enumerate(train_reader()):
            loss, acc, weight = exe.run(fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, batch_acc, batch_size])
            accuracy.add(value=acc, weight=weight)

            if batch_id % 100 == 0 :
                print("Pass {0}, batch {1}, loss {2}, acc {3}".format(
                    pass_id, batch_id, loss[0], acc[0]))


        # 测试模型
        test_accuracy.reset()
        for data in test_reader():
            loss, acc, weight = exe.run(inference_program,
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, batch_acc, batch_size])
            test_accuracy.add(value=acc, weight=weight)

        # 输出相关日志
        pass_acc = accuracy.eval()
        test_pass_acc = test_accuracy.eval()
        print("End pass {0}, train_acc {1}, test_acc {2}".format(
            pass_id, pass_acc, test_pass_acc))


    fluid.io.save_params(
        exe, dirname=dirname, main_program=fluid.default_main_program())
    print('train mnist done')



def make_mlp_model_plus(dirname):
    """
    Train the cnn model on mnist datasets
    """

    pkl_file = open(adversarial_examples_pkl, 'rb')
    (x,y) = pickle.load(pkl_file)

    """
    Train the cnn model on mnist datasets
    """
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    logits = mnist_mlp_model(img)
    cost = fluid.layers.cross_entropy(input=logits, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    optimizer = fluid.optimizer.Adam(learning_rate=0.01)
    optimizer.minimize(avg_cost)

    # 每个batch计算的时候能取到当前batch里面样本的个数，从而来求平均的准确率
    batch_size = fluid.layers.create_tensor(dtype='int64')
    #print "print batch_size=%d" % batch_size
    batch_acc = fluid.layers.accuracy(
        input=logits, label=label, total=batch_size)



    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

    # set to True if training with GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(fluid.default_startup_program())

    # 测试程序
    inference_program = fluid.default_main_program().clone(for_test=True)

    accuracy = fluid.average.WeightedAverage()
    test_accuracy = fluid.average.WeightedAverage()
    # 开始训练，使用循环的方式来指定训多少个Pass
    for pass_id in range(PASS_NUM):
        # 从训练数据中按照一个个batch来读取数据
        accuracy.reset()
        for batch_id, data in enumerate(train_reader()):
            loss, acc, weight = exe.run(fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, batch_acc, batch_size])
            accuracy.add(value=acc, weight=weight)

            if batch_id % 10 == 0 :
                print("[MNIST]Pass {0}, batch {1}, loss {2}, acc {3}".format(
                    pass_id, batch_id, loss[0], acc[0]))

        # 使用对抗样本继续训练

        for index, x0 in enumerate(x):

            data=[[x0,y[index]]]

            loss, acc, weight = exe.run(fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, batch_acc, batch_size])
            accuracy.add(value=acc, weight=weight)

            if index % 10 == 0 :
                print("[ADV]Pass {0}, batch {1}, loss {2}, acc {3}".format(
                    pass_id, index, loss[0], acc[0]))


        # 测试模型
        test_accuracy.reset()
        for data in test_reader():
            loss, acc, weight = exe.run(inference_program,
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, batch_acc, batch_size])
            test_accuracy.add(value=acc, weight=weight)

        # 输出相关日志
        pass_acc = accuracy.eval()
        test_pass_acc = test_accuracy.eval()
        print("[MNIST+ADV]End pass {0}, train_acc {1}, test_acc {2}".format(
            pass_id, pass_acc, test_pass_acc))


    fluid.io.save_params(
        exe, dirname=dirname, main_program=fluid.default_main_program())
    print('train mnist done')






def get_adversarial_examples_from_modela():
    """
    Advbox demo which demonstrate how to use advbox.
    """

    IMG_NAME = 'img'
    LABEL_NAME = 'label'

    #记录对抗样本
    x=[]
    y=[]

    img = fluid.layers.data(name=IMG_NAME, shape=[1, 28, 28], dtype='float32')
    # gradient should flow
    img.stop_gradient = False
    label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')
    logits = mnist_mlp_model(img)
    cost = fluid.layers.cross_entropy(input=logits, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # set to True if training with GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    BATCH_SIZE = 1
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.test(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    fluid.io.load_params(
        exe, modela_path, main_program=fluid.default_main_program())

    # advbox demo
    m = PaddleModel(
        fluid.default_main_program(),
        IMG_NAME,
        LABEL_NAME,
        logits.name,
        avg_cost.name, (-1, 1),
        channel_axis=1)
    attack = DeepFoolAttack(m)
    attack_config = {"iterations": 100, "overshoot": 9}

    # use test data to generate adversarial examples
    total_count = 0
    fooling_count = 0

    #使用训练集产生对抗样本
    for data in train_reader():
    #for data in test_reader():
        total_count += 1
        adversary = Adversary(data[0][0], data[0][1])

        # DeepFool non-targeted attack
        adversary = attack(adversary, **attack_config)

        # DeepFool targeted attack
        # tlabel = 0
        # adversary.set_target(is_targeted_attack=True, target_label=tlabel)
        # adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1

            x.append(adversary.adversarial_example)
            y.append(adversary.adversarial_label)

            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (data[0][1], adversary.adversarial_label, total_count))

        else:
            print('attack failed, original_label=%d, count=%d' %
                  (data[0][1], total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break
    print("deelfool attack done")

    z=(x,y)

    output = open(adversarial_examples_pkl, 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(z, output)

    return z


def main():
    #训练样本
    #make_mlp_model_new(modela_path)
    #生成对抗样本 仅针对训练数据生成对抗样本
    #(x,y)=get_adversarial_examples_from_modela()

    #重新训练模型
    make_mlp_model_plus(modelb_path)



if __name__ == '__main__':
    main()

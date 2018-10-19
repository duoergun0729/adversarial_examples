"""
CNN on mnist data using fluid api of paddlepaddle
"""
import paddle.v2 as paddle
import paddle.fluid as fluid

# Plot data
from paddle.v2.plot import Ploter

import matplotlib
matplotlib.use('Agg')

step = 0

def mnist_cnn_model(img):
    """
    Mnist cnn model

    Args:
        img(Varaible): the input image to be recognized

    Returns:
        Variable: the label prediction
    """
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        num_filters=20,
        filter_size=5,
        pool_size=2,
        pool_stride=2,
        act='relu')

    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        num_filters=50,
        filter_size=5,
        pool_size=2,
        pool_stride=2,
        act='relu')
    fc = fluid.layers.fc(input=conv_pool_2, size=50, act='relu')

    logits = fluid.layers.fc(input=fc, size=10, act='softmax')
    return logits

def mnist_mlp_model(img):
    """
    Mnist mlp model

    Args:
        img(Varaible): the input image to be recognized

    Returns:
        Variable: the label prediction
    """

    fc1 = fluid.layers.fc(input=img, size=512, act='relu')

    fc2 = fluid.layers.fc(input=fc1, size=512, act='relu')

    logits = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return logits


def make_cnn_model(dirname):
    """
    Train the cnn model on mnist datasets
    """
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    logits = mnist_cnn_model(img)
    cost = fluid.layers.cross_entropy(input=logits, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    optimizer = fluid.optimizer.Adam(learning_rate=0.01)
    optimizer.minimize(avg_cost)

    batch_size = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=logits, label=label, total=batch_size)

    BATCH_SIZE = 50
    PASS_NUM = 3
    ACC_THRESHOLD = 0.98
    LOSS_THRESHOLD = 10.0
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    # use CPU
    place = fluid.CPUPlace()
    # use GPU
    # place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(fluid.default_startup_program())

    pass_acc = fluid.average.WeightedAverage()
    for pass_id in range(PASS_NUM):
        pass_acc.reset()
        for data in train_reader():
            loss, acc, b_size = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost, batch_acc, batch_size])
            pass_acc.add(value=acc, weight=b_size)
            pass_acc_val = pass_acc.eval()[0]
            print("pass_id=" + str(pass_id) + " acc=" + str(acc[0]) +
                  " pass_acc=" + str(pass_acc_val))
            if loss < LOSS_THRESHOLD and pass_acc_val > ACC_THRESHOLD:
                # early stop
                break

        print("pass_id=" + str(pass_id) + " pass_acc=" + str(pass_acc.eval()[
            0]))
    fluid.io.save_params(
        exe, dirname=dirname, main_program=fluid.default_main_program())
    print('train mnist cnn model done')

def make_mlp_model(dirname):
    """
    Train the mlp model on mnist datasets
    """
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    logits = mnist_mlp_model(img)

    cost = fluid.layers.cross_entropy(input=logits, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    optimizer = fluid.optimizer.Adam(learning_rate=0.01)
    optimizer.minimize(avg_cost)

    batch_size = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=logits, label=label, total=batch_size)

    BATCH_SIZE = 50
    PASS_NUM = 5
    ACC_THRESHOLD = 0.98
    LOSS_THRESHOLD = 10.0
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    # use CPU
    place = fluid.CPUPlace()
    # use GPU
    # place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(fluid.default_startup_program())

    pass_acc = fluid.average.WeightedAverage()
    for pass_id in range(PASS_NUM):
        pass_acc.reset()
        for data in train_reader():
            loss, acc, b_size = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost, batch_acc, batch_size])
            pass_acc.add(value=acc, weight=b_size)
            pass_acc_val = pass_acc.eval()[0]
            print("pass_id=" + str(pass_id) + " acc=" + str(acc[0]) +
                  " pass_acc=" + str(pass_acc_val))
            if loss < LOSS_THRESHOLD and pass_acc_val > ACC_THRESHOLD:
                # early stop
                break

        print("pass_id=" + str(pass_id) + " pass_acc=" + str(pass_acc.eval()[
            0]))
    fluid.io.save_params(
        exe, dirname=dirname, main_program=fluid.default_main_program())



    print('train mnist mlp model done')


def make_cnn_model_visualization(dirname):
    """
    Train the cnn model on mnist datasets
    """

    batch_size = 64
    num_epochs = 5

    use_cuda = 1


    def optimizer_program():
        return fluid.optimizer.Adam(learning_rate=0.001)

    def train_program():
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')


        predict =  mnist_cnn_model(img)

        # Calculate the cost from the prediction and label.
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(cost)
        acc = fluid.layers.accuracy(input=predict, label=label)
        return [avg_cost, acc]

    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
        batch_size=batch_size)

    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=batch_size)

      # set to True if training with GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    trainer = fluid.Trainer(
        train_func=train_program, place=place, optimizer_func=optimizer_program)

    # Save the parameter into a directory. The Inferencer can load the parameters from it to do infer
    params_dirname = dirname

    lists = []

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            if event.step % 100 == 0:
                # event.metrics maps with train program return arguments.
                # event.metrics[0] will yeild avg_cost and event.metrics[1] will yeild acc in this example.
                print "step %d, epoch %d, Cost %f Acc %f " % (event.step, event.epoch,event.metrics[0],event.metrics[1])

        if isinstance(event, fluid.EndEpochEvent):
            avg_cost, acc = trainer.test(
                reader=test_reader, feed_order=['img', 'label'])

            print("Test with Epoch %d, avg_cost: %s, acc: %s" %
                  (event.epoch, avg_cost, acc))

            # save parameters
            print "save_params"
            trainer.save_params(params_dirname)
            lists.append((event.epoch, avg_cost, acc))


    # Train the model now
    trainer.train(
        num_epochs=num_epochs,
        event_handler=event_handler,
        reader=train_reader,
        feed_order=['img', 'label'])

    # find the best pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
    print 'The classification accuracy is %.2f%%' % (float(best[2]) * 100)



def make_mlp_model_visualization(dirname):
    """
    Train the cnn model on mnist datasets
    """

    batch_size = 64
    num_epochs = 10

    use_cuda = 1


    def optimizer_program():
        return fluid.optimizer.Adam(learning_rate=0.001)

    def train_program():
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')


        predict =  mnist_mlp_model(img)

        # Calculate the cost from the prediction and label.
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(cost)
        acc = fluid.layers.accuracy(input=predict, label=label)
        return [avg_cost, acc]

    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
        batch_size=batch_size)

    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=batch_size)

      # set to True if training with GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    trainer = fluid.Trainer(
        train_func=train_program, place=place, optimizer_func=optimizer_program)

    # Save the parameter into a directory. The Inferencer can load the parameters from it to do infer
    params_dirname = dirname

    lists = []

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            if event.step % 100 == 0:
                # event.metrics maps with train program return arguments.
                # event.metrics[0] will yeild avg_cost and event.metrics[1] will yeild acc in this example.
                print "step %d, epoch %d, Cost %f Acc %f " % (event.step, event.epoch,event.metrics[0],event.metrics[1])

        if isinstance(event, fluid.EndEpochEvent):
            avg_cost, acc = trainer.test(
                reader=test_reader, feed_order=['img', 'label'])

            print("Test with Epoch %d, avg_cost: %s, acc: %s" %
                  (event.epoch, avg_cost, acc))

            # save parameters
            print "save_params"
            trainer.save_params(params_dirname)
            lists.append((event.epoch, avg_cost, acc))


    # Train the model now
    trainer.train(
        num_epochs=num_epochs,
        event_handler=event_handler,
        reader=train_reader,
        feed_order=['img', 'label'])

    # find the best pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
    print 'The classification accuracy is %.2f%%' % (float(best[2]) * 100)

def main():
    #make_cnn_model_visualization('./mnist_blackbox/cnn/')
    #make_mlp_model('./mnist_blackbox/mlp')
    make_mlp_model_visualization('./mnist_blackbox/mlp/')


if __name__ == '__main__':
    main()

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
    print('train mnist mlp model done')


def make_cnn_model_visualization(dirname):
    """
    Train the cnn model on mnist datasets
    """

    BATCH_SIZE = 1
    PASS_NUM = 3
    ACC_THRESHOLD = 0.98
    LOSS_THRESHOLD = 10.0

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.test(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    def train_program():

        img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        logits = mnist_cnn_model(img)
        cost = fluid.layers.cross_entropy(input=logits, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        accuracy = fluid.layers.accuracy(input=logits, label=label)

        return [avg_cost,accuracy]

    def optimizer_program():
        return fluid.optimizer.Adam(learning_rate=0.01)



    # use CPU
    place = fluid.CPUPlace()
    # use GPU
    # place = fluid.CUDAPlace(0)


    trainer = fluid.Trainer(
        train_func=train_program, place=place, optimizer_func=optimizer_program)

    feed_order = ['img', 'label']

    # Specify the directory path to save the parameters
    params_dirname = dirname

    # Plot data
    from paddle.v2.plot import Ploter
    train_title = "Train cost"
    test_title = "Test cost"
    plot_cost = Ploter(train_title, test_title)


    # event_handler to print training and testing info
    def event_handler_plot(event):
        if isinstance(event, fluid.EndStepEvent):
            if event.step % 10 == 0:  # every 10 batches, record a test cost
                avg_cost, accuracy = trainer.test(
                    reader=test_reader, feed_order=feed_order)

                print('\nTrain with Pass {0}, Loss {1:2.2}, Acc {2:2.2}'.format(
                    event.epoch, avg_cost, accuracy))
                #plot_cost.plot()


            # We can save the trained parameters for the inferences later
            if params_dirname is not None:
                trainer.save_params(params_dirname)

        if isinstance(event, fluid.EndEpochEvent):
            avg_cost, accuracy = trainer.test(
                reader=test_reader, feed_order=feed_order)

            print('\nTest with Pass {0}, Loss {1:2.2}, Acc {2:2.2}'.format(
                event.epoch, avg_cost, accuracy))
            if params_dirname is not None:
                trainer.save_params(params_dirname)

    # The training could take up to a few minutes.
    trainer.train(
        reader=train_reader,
        num_epochs=PASS_NUM,
        event_handler=event_handler_plot,
        feed_order=feed_order)





def main():
    make_cnn_model_visualization('./mnist_blackbox/cnn')
    #make_mlp_model('./mnist_blackbox/mlp')


if __name__ == '__main__':
    main()

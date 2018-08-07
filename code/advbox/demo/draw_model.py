# -- coding: utf-8 --

import mxnet as mx  

def draw_mnist_cnn():
    '''
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
    '''

    #网络定义
    data = mx.symbol.Variable('mnist')

    conv1= mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
    act1= mx.sym.Activation(data=conv1, act_type="relu")
    pool1= mx.sym.Pooling(data=act1, pool_type="max", kernel=(2,2), stride=(2,2))

    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    act2 = mx.sym.Activation(data=conv2, act_type="relu")
    pool2 = mx.sym.Pooling(data=act2, pool_type="max", kernel=(2, 2), stride=(2, 2))

    fc1 = mx.symbol.FullyConnected(data=pool2, name='fc1', num_hidden=50)
    act3 = mx.sym.Activation(data=fc1, act_type="relu")

    fc2 = mx.symbol.FullyConnected(data=act3, name='fc2', num_hidden=10)

    mlp = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    # 网络可视化
    mx.viz.plot_network(mlp,node_attrs={"shape":'oval',"fixedsize":'false'}).view()


def draw_mnist_mlp():
    '''
    fc1 = fluid.layers.fc(input=img, size=512, act='relu')
    fc2 = fluid.layers.fc(input=fc1, size=512, act='relu')
    logits = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return logits
    '''
    # 网络定义
    data = mx.symbol.Variable('mnist')
    fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=512)
    act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type='relu')
    fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=512)
    act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type='relu')
    fc3 = mx.symbol.FullyConnected(data=act2, name='fc3', num_hidden=10)
    mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')

    # 网络可视化
    mx.viz.plot_network(mlp,node_attrs={"shape":'oval',"fixedsize":'false'}).view()

if __name__ == '__main__':
    draw_mnist_mlp()
    draw_mnist_cnn()

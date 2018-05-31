# -*- coding: utf-8 -*-

import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
from PIL import Image

"""
pip install keras==2.1.5
pip install tensorflow==1.4
pip install Pillow
pip install h5py
"""


def demo4():
    model = inception_v3.InceptionV3()

    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    object_type_to_fake = 859


    img = image.load_img("../picture/pig.jpg", target_size=(299, 299))
    original_image = image.img_to_array(img)

    original_image /= 255.
    original_image -= 0.5
    original_image *= 2.

    original_image = np.expand_dims(original_image, axis=0)


    max_change_above = original_image + 0.01
    max_change_below = original_image - 0.01


    hacked_image = np.copy(original_image)


    learning_rate = 0.1


    cost_function = model_output_layer[0, object_type_to_fake]


    gradient_function = K.gradients(cost_function, model_input_layer)[0]


    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

    cost = 0.0

    index=1
    while cost < 0.60:

        cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])


        hacked_image += gradients * learning_rate
        #print gradients

        hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
        hacked_image = np.clip(hacked_image, -1.0, 1.0)

        print("batch:{} Cost: {:.8}%".format(index,cost * 100))
        index+=1

    img = hacked_image[0]
    img /= 2.
    img += 0.5
    img *= 255.

    #im = Image.fromarray(img.astype(np.uint8))
    #im.save("hacked-pig-image.png")


if __name__ == '__main__':
    demo4()

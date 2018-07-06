# -*- coding: utf-8 -*-

from tensorflow.core.framework.graph_pb2 import *
import numpy as np
import tensorflow as tf

import sys

DeepSpeech_PATH="/Volumes/maidou/dataset/audio_adversarial_examples/DeepSpeech/"

models_file="/Volumes/maidou/dataset/audio_adversarial_examples/models/output_graph.pb"

wav_file="case1.wav"

alphabet_file="/Volumes/maidou/dataset/audio_adversarial_examples/models/alphabet.txt"


sys.path.append(DeepSpeech_PATH)

from util.audio import audiofile_to_input_vector
from util.text import ctc_label_dense_to_sparse

# Okay, so this is ugly. We don't want DeepSpeech to crash
# when we haven't built the language model.
# So we're just going to monkeypatch TF and make it a no-op.
# Sue me.
tf.load_op_library = lambda x: x
import DeepSpeech as DeepSpeech

graph_def = GraphDef()
loaded = graph_def.ParseFromString(open(models_file,"rb").read())

with tf.Graph().as_default() as graph:
    new_input = tf.placeholder(tf.float32, [None, None, None],
                               name="new_input")
    # Load the saved .pb into the current graph to let us grab
    # access to the weights.
    logits, = tf.import_graph_def(
        graph_def,
        input_map={"input_node:0": new_input},
        return_elements=['logits:0'],
        name="newname",
        op_dict=None,
        producer_op_list=None
    )

    # Now let's dump these weights into a new copy of the network.
    with tf.Session(graph=graph) as sess:
        # Sample sentetnce, to make sure we've done it right
        mfcc = audiofile_to_input_vector(wav_file, 26, 9)

        # Okay, so this is ugly again.
        # We just want it to not crash.
        tf.app.flags.FLAGS.alphabet_config_path = alphabet_file
        DeepSpeech.initialize_globals()
        logits2 = DeepSpeech.BiRNN(new_input, [len(mfcc)], [0]*10)

        # Here's where all the work happens. Copy the variables
        # over from the .pb to the session object.
        for var in tf.global_variables():
            sess.run(var.assign(sess.run('newname/'+var.name)))

        # Test to make sure we did it right.
        res = (sess.run(logits, {new_input: [mfcc],
                                     'newname/input_lengths:0': [len(mfcc)]}).flatten())
        res2 = (sess.run(logits2, {new_input: [mfcc]})).flatten()
        print('This value should be small',np.sum(np.abs(res-res2)))



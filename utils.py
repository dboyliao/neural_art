#!/usr/bin/env python3
# -*- coding:utf8 -*-
from functools import reduce
from PIL import Image
import tensorflow as tf
import numpy as np


def load_graph_pb(path, graph=None):
    """
    Arguemnts
    =========
    - path <str>: the path of the pb file
    """
    if graph is None:
        graph = tf.get_default_graph()
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(path, 'rb') as fid:
        graph_def.ParseFromString(fid.read())

    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return graph


def gram_matrix(activation, name="Gram_Matrix"):
    """
    args
    ====
    - activation: 1 by W by H by C tensor.
    """
    with tf.name_scope(name):
        with tf.name_scope("preprocess"):
            activation = tf.transpose(activation, perm=[3, 0, 1, 2])
            M = reduce(lambda acc, x: acc * x, activation.shape.as_list()[2:], 1)
            F = tf.reshape(activation, shape=[-1, M], name="F")
        G = tf.matmul(F, tf.transpose(F), name="gram_matrix")
    return G


def read_image(path, size=None, img_format=None):
    pil_image = Image.open(path)
    if size:
        pil_image = pil_image.resize(size)
    if img_format:
        pil_image = pil_image.convert(img_format)
    return np.array(pil_image)


def tensor_name(name, scope=None):
    if scope:
        return "{}/{}:0".format(scope, name)
    return "{}:0".format(name)

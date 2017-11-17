# -*- coding:utf8 -*-
# pylint: disable=C0301
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
from utils import tensor_name, gram_matrix


def TV_loss(images, ord=2.0, name="SmoothTotalVariation"):
    """Smoothed TV loss
    """
    ndims = images.shape.ndims
    if ndims == 3:
        sum_axis = None
    elif ndims == 4:
        sum_axis = [1, 2, 3]
    else:
        raise ValueError("'images' should be 3 or 4-dimensional tensor")

    with tf.name_scope(name, [images]):
        diffx = images[:, 1:, :] - images[:, :-1, :]
        diffy = images[:, :, 1:] - images[:, :, :-1]
        norm = tf.pow(diffx[:, :, :-1]**2 + diffy[:, :-1, :]**2, ord/2)
    return tf.reduce_sum(norm, name="sum_norm", axis=sum_axis)


def build_artist(vgg_net, alpha, beta,
                 content_layer_name,
                 style_layer_names,
                 style_weights,
                 use_smooth_tv=False):
    artist_net = tf.Graph()
    ret_tensors = {}
    with artist_net.as_default():
        # setup hyper parameters
        tf_alpha = tf.constant(alpha, dtype=tf.float32, name="alpha")
        tf_beta_tv = tf.constant(beta, dtype=tf.float32, name="beta_tv")
        tf_style_weights = [tf.constant(w, dtype=tf.float32, name="style_weight{}".format(i))
                            for i, w in enumerate(style_weights, 1)]

        # the output image we want
        # Note that the output image will be noramlized to [0, 1]
        tf_output_image = tf.Variable(np.random.randn(1, 224, 224, 3), 
                                      dtype=tf.float32,
                                      name="output_image")
        ret_tensors["output_image"] = tf_output_image

        with tf.name_scope("content_image"):
            tf_content_image = tf.placeholder(tf.float32,
                                              shape=[1, 224, 224, 3],
                                              name="image")
        ret_tensors["content_image"] = tf_content_image
        with tf.name_scope("style_image"):
            tf_style_image = tf.placeholder(tf.float32,
                                            shape=[1, 224, 224, 3],
                                            name="image")
        ret_tensors["style_image"] = tf_style_image

        # import three vgg graphs in to artist graph
        # one for constructed image, one for content representation, one for style
        ## content image
        sub_vgg_graph_def = graph_util.extract_sub_graph(vgg_net.as_graph_def(),
                                                         [content_layer_name])
        tf.import_graph_def(sub_vgg_graph_def, 
                            name="vgg_content",
                            input_map={"input_rgb": tf_content_image})
        ## style image
        sub_vgg_graph_def = graph_util.extract_sub_graph(vgg_net.as_graph_def(),
                                                         style_layer_names)
        tf.import_graph_def(sub_vgg_graph_def,
                            name="vgg_style",
                            input_map={"input_rgb": tf_style_image})

        ## output image
        all_names = [content_layer_name] + style_layer_names
        sub_vgg_graph_def = graph_util.extract_sub_graph(vgg_net.as_graph_def(),
                                                         all_names)
        tf.import_graph_def(sub_vgg_graph_def,
                            name="vgg_fit",
                            input_map={"input_rgb": tf_output_image})
        del sub_vgg_graph_def

        ## loss
        ret_losses = {}
        with tf.name_scope("loss"):
            ## content loss
            with tf.name_scope("content_loss"):
                content_layer = artist_net.get_tensor_by_name(tensor_name(content_layer_name,
                                                                          "vgg_content"))
                content_act = artist_net.get_tensor_by_name(tensor_name(content_layer_name,
                                                                        "vgg_fit"))
            with tf.control_dependencies([content_layer, content_act]):
                loss_content = tf.multiply(0.5,
                                           tf.reduce_sum(tf.squared_difference(content_layer, content_act)),
                                           name="square_sum_loss")
            ret_losses["content_loss"] = loss_content
            ## style loss
            loss_style = None
            with tf.name_scope("style_loss"):
                for weight, name in zip(tf_style_weights, style_layer_names):
                    style_layer = artist_net.get_tensor_by_name(tensor_name(name, "vgg_style"))
                    G1 = gram_matrix(style_layer, "GramMatrix1_{}".format(name))
                    style_act = artist_net.get_tensor_by_name(tensor_name(name, "vgg_fit"))
                    G2 = gram_matrix(style_act, "GramMatrix2_{}".format(name))
                    shape = style_layer.shape.as_list()
                    N = shape[3]
                    M = shape[1] * shape[2]
                    with tf.control_dependencies([style_layer, style_act]):
                        dloss = tf.multiply(weight / tf.constant((2 * N * M)**2, dtype=tf.float32),
                                            tf.reduce_sum(tf.squared_difference(G1, G2)),
                                            name="loss_{}".format(name))
                    if loss_style is None:
                        loss_style = dloss
                    else:
                        loss_style += dloss
            ret_losses["style_loss"] = loss_style
            with tf.name_scope("TV_loss"):
                if use_smooth_tv:
                    loss_tv = TV_loss(tf_output_image)[0]
                else:
                    loss_tv = tf.image.total_variation(tf_output_image, name="loss")[0]
            ret_losses["tv_loss"] = loss_tv
            ## total loss
            loss = tf.reduce_sum([tf_alpha * loss_content,
                                 tf_beta_tv * loss_tv,
                                 loss_style],
                                 name="total_loss")
            ret_losses["total_loss"] = loss

        with tf.name_scope("optimization"):
            train_op = tf.train.AdamOptimizer(20.0).minimize(loss)
        saver = tf.train.Saver()
        return artist_net, ret_tensors, ret_losses, train_op, saver

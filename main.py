#!/usr/bin/env python3
# -*- coding:utf8 -*-
import argparse
import time
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import read_image, load_graph_pb
from artist import build_artist


def main(alpha, beta,
         content_layer,
         style_layers, style_weights,
         use_smooth_tv,
         vgg_path,
         content_image,
         style_image,
         num_iteration,
         log_every_step,
         output_name):
    """main function
    """
    assert len(style_layers) == len(style_weights), \
        "the number of layers and weights do no match"
    vgg_net = load_graph_pb(vgg_path)
    artist_net, ret_tensors, ret_losses, train_op, saver = build_artist(vgg_net,
                                                                        alpha, beta,
                                                                        content_layer,
                                                                        style_layers,
                                                                        style_weights,
                                                                        use_smooth_tv)
    tf_output_image = ret_tensors["output_image"]
    tf_content_image = ret_tensors["content_image"]
    tf_style_image = ret_tensors["style_image"]

    loss = ret_losses["total_loss"]
    loss_content = ret_losses["content_loss"]
    loss_style = ret_losses["style_loss"]
    loss_tv = ret_losses["tv_loss"]

    content_image = read_image(content_image, (224, 224), "RGB")
    style_image = read_image(style_image, (224, 224), "RGB")

    try:
        with tf.Session(graph=artist_net) as sess:
            tf.global_variables_initializer().run()
            print("All variables inititialized")
            feed_dict = {tf_content_image: content_image[None,:],
                         tf_style_image: style_image[None,:]}
            tl, cl, sl, tvl, alpha_, beta_ = sess.run([loss, loss_content,
                                                       loss_style, loss_tv], 
                                                       feed_dict=feed_dict)
            print("Initial losses:", tl)
            print("  Content loss:", cl)
            print("  Style loss:", sl)
            print("  TV loss:", tvl)
            durations = []
            for step in range(1, num_iteration + 1):
                start_time = time.time()
                l, cl, sl, tvl, _ = sess.run([loss, loss_content, loss_style, 
                                              loss_tv, train_op], 
                                              feed_dict=feed_dict)
                end_time = time.time()
                durations.append(end_time-start_time)
                if step % log_every_step == 0:
                    print("step:", step)
                    print("totol loss:", l)
                    print("content loss:", cl)
                    print("style loss:", sl)
                    print("TV loss:", tvl)
                    print("mean duration: {:.4f} sec".format(np.mean(durations)))
                    if np.isnan(l):
                        print("nan detected, breaking training loop")
                        break
                    chkp_path = saver.save(sess, save_path, step)
                    durations = []

            out_image = tf_output_image.eval()[0]
        # saving image
        out_image = np.clip(out_image, 0, 255).astype(np.uint8)
        Image.fromarray(out_image).resize((512, 512)).save(output_name)
        print("Saving image: {}".format(output_name))
    except BaseException as e:
        print("interrupted...")
        print(type(e))
        print(e)


def _list_arg(to_type=None, sep=','):
    def parse(arg_str):
        l = arg_str.strip().split(sep)
        if to_type is not None:
            l = list(map(to_type, l))
        return l
    return parse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # hyper parameters
    parser.add_argument("--content-layer", dest="content_layer",
                        default="conv4_2/Relu",
                        help="layer for content loss")
    parser.add_argument("--style-layers", dest="style_layers",
                        help="style layer names, seperated by ',' (defaults: %(default)s)",
                        default="conv1_1/Relu,conv2_1/Relu, conv3_1/Relu,conv4_1/Relu,conv5_1/Relu",
                        type=_list_arg())
    parser.add_argument("--style-weights", dest="style_weights",
                        help="style weights, seperated by ',' (defaults: %(default)s)",
                        default="1,1,1,1,1",
                        type=_list_arg(to_type=float))
    parser.add_argument("--alpha", dest="alpha",
                        help="alpha, for content loss (default: %(default).1E)",
                        default=1e3, type=float)
    parser.add_argument("--beta", dest="beta",
                        help="beta, for total variation loss (default: %(default).1E)",
                        default=8e11, type=float)
    parser.add_argument("--use-smooth-tv", action="store_true",
                        dest="use_smooth_tv",
                        help="use smoothed version TV loss")
    parser.add_argument("--save-path", dest="save_path",
                        help="session save path")
    # training parameters
    parser.add_argument("--vgg-path", dest="vgg_path",
                        help="path of the pb file of vgg net (default: %(default)s)",
                        default="vgg19.pb")
    parser.add_argument("--content-image", dest="content_image",
                        help="path of the content image", required=True)
    parser.add_argument("--style-image", dest="style_image",
                        help="path of the style image", required=True)
    parser.add_argument("--iters", dest="num_iteration",
                        type=int, default=500,
                        help="number of iterations for optimization (default: %(default)d)")
    parser.add_argument("--log-steps", dest="log_every_step",
                        type=int, default=100,
                        help="duration of logging steps (default: %(default)d)")
    parser.add_argument("-o", "--output-image", dest="output_name",
                        default="styled_image.png",
                        help="output image name (default: %(default)s)")
    args = vars(parser.parse_args())
    main(**args)

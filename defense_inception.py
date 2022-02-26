"""Implementation of sample defense.

This defense loads inception resnet v2 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import os

import random

import numpy as np
from scipy.misc import imread

import tensorflow as tf

from tensorflow.contrib.slim.nets import inception
from nets import inception_v4,inception_v3

import logging
logging.getLogger('tensorflow').disabled = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '/home/lthpc/workspace/zhangshudong/adve/checkpoints/inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '/home/lthpc/workspace/zhangshudong/adve/attacks/white_box/auto_attack/InceptionV3/apgd-ce', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', 'label', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 50, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'image_resize', 331, 'Resize of image size.')

FLAGS = tf.flags.FLAGS



def  padding_layer_iyswim(inputs, shape, name=None):
    h_start = shape[0]
    w_start = shape[1]
    output_short = shape[2]
    input_shape = tf.shape(inputs)

    input_short = tf.reduce_min(input_shape[1:3])
    input_long = tf.reduce_max(input_shape[1:3])

    output_long = tf.to_int32(tf.ceil(
        1. * tf.to_float(output_short) * tf.to_float(input_long) / tf.to_float(input_short)))
    output_height = tf.to_int32(input_shape[1] >= input_shape[2]) * output_long +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_short
    output_width = tf.to_int32(input_shape[1] >= input_shape[2]) * output_short +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_long
    return tf.pad(inputs, tf.to_int32(tf.stack([[0, 0], [h_start, output_height - h_start - input_shape[1]], [w_start, output_width - w_start - input_shape[2]], [0, 0]])), name=name)

def get_file(path):
  a = []
  sec_dirs_list = [i[0] for i in os.walk(path)][1:]
  for sec_dir_path in sec_dirs_list:
    for file in os.listdir(sec_dir_path):
      a.append(os.path.join(sec_dir_path, file))
  print(len(a))
  return a

def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]

    # filepaths = get_file(input_dir)

    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    # for filepath in filepaths:
        with tf.gfile.Open(filepath,'rb') as f:
            # image = imread(f, mode='RGB').astype(np.float) / 255.0
            image = np.array(Image.open(f).resize((299, 299)).convert('RGB')).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images



def main(_):

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001
    itr = 1

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        img_resize_tensor = tf.placeholder(tf.int32, [2])
        x_input_resize = tf.image.resize_images(x_input, img_resize_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        shape_tensor = tf.placeholder(tf.int32, [3])
        padded_input = padding_layer_iyswim(x_input_resize, shape_tensor)
        # 330 is the last value to keep 8*8 output, 362 is the last value to keep 9*9 output, stride = 32
        padded_input.set_shape(
            (FLAGS.batch_size, FLAGS.image_resize, FLAGS.image_resize, 3))

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            _, end_points = inception_v3.inception_v3(
                padded_input, num_classes=num_classes, is_training=False)
            aux_pred_ = tf.nn.softmax(end_points['AuxLogits'])

        predicted_labels = tf.argmax(end_points['Predictions'], 1)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path,
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    final_preds = np.zeros(
                        [FLAGS.batch_size, num_classes, itr])
                    for j in range(itr):
                        if np.random.randint(0, 2, size=1) == 1:
                            images = images[:, :, ::-1, :]
                        resize_shape_ = np.random.randint(310, 331)

                        pred, aux_pred = sess.run([end_points['Predictions'], aux_pred_],
                                                        feed_dict={x_input: images, img_resize_tensor: [resize_shape_]*2,
                                                                   shape_tensor: np.array([random.randint(0, FLAGS.image_resize - resize_shape_), random.randint(0, FLAGS.image_resize - resize_shape_), FLAGS.image_resize])})


                        final_preds[..., j] = pred + 0.4 * aux_pred
                    final_probs = np.sum(final_preds, axis=-1)
                    labels = np.argmax(final_probs, 1)
                    for filename, label in zip(filenames, labels):
                        out_file.write('{0},{1}\n'.format(filename, label))

    fr = open('/home/lthpc/workspace/zhangshudong/adve/test_adve/inception_v3/1000label.txt', 'r')
    dic = {}
    dic1 = {}
    keys = []
    x = 0
    for line in fr:
        a = line.strip().split(':')
        # print(a)  # a[0] represent picture name      a[1] represent picture label
        dic[a[0]] = a[1]  # image name -> str label
        dic1[a[0]] = x  # hang -> str label
        x += 1
        keys.append(a[0])
    fr.close()
    count = 0
    num = 0
    o = open(FLAGS.output_file,'r')
    for line in o:
        name = line.strip().split(',')
        name_ = name[0].split('_')[0]
        num+=1
        if dic1[name_] == int(name[-1])-1:
            count+=1
    print(num)
    num = float(num)
    with open('result.txt','a') as r:
        r.write(str(FLAGS.input_dir.split('/')[-1])+' '+str((count/num)*100)+'\n')
    print((count/num)*100)

if __name__ == '__main__':
    tf.app.run()

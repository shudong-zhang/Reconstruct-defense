from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from lib.ops import *
import collections
import os
import math
import scipy.misc as sic
import numpy as np

from tensorflow.contrib.slim.nets import inception
from lib.inception_preprocessing import preprocess_image

def get_image(noisy_path,ori_path,count):
    a = []
    noisy_image_path = []
    noisy_image = []
    ori_image = []

    for root, dirs, files in os.walk(noisy_path):
        a.append(root)
    for i in a:
        if i.split("/")[-1].startswith('n'):
            noisy_image_path.append(i)

    i = 0
    while i < len(noisy_image_path):
        # print(len(a) - 1)
        # print('i:', i)
        if 'IncResV2' in noisy_image_path[i] or 'ResV2' in noisy_image_path[i]:
            noisy_image_path.remove(str(noisy_image_path[i]))
            if i != 0:
                i = i - 1
        else:
            i = i + 1

    for i in noisy_image_path:

        for image in os.listdir(i)[0:count]:
            noisy_image.append(os.path.join(i, image))



    noisy_image = sorted(noisy_image)

    for i in noisy_image:
        split_path = i.split('/')
        ori_image.append(os.path.join(os.path.join(ori_path, split_path[-2]), split_path[-1]))
    return noisy_image, ori_image

def get_test_image(noisy_path,model_name):
    noisy_image_path = []
    noisy_image = []
    a =[i[0] for i in os.walk(noisy_path)]
    for i in a:
        if i.split("/")[-1].startswith('n') and str(model_name) in i:
            noisy_image_path.append(i)
    for i in noisy_image_path:
        for image in os.listdir(i):
            noisy_image.append(os.path.join(i,image))
    noisy_image = sorted(noisy_image)
    return noisy_image

# Define the dataloader
def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        # Define the returned data batches
        Data = collections.namedtuple('Data', 'paths_adv, paths_nat, inputs, targets, image_count, steps_per_epoch')

        #Check the input directory
        if (FLAGS.input_dir_adv == 'None') or (FLAGS.input_dir_nat == 'None'):
            raise ValueError('Input directory is not provided')

        if (not os.path.exists(FLAGS.input_dir_adv)) or (not os.path.exists(FLAGS.input_dir_nat)):
            raise ValueError('Input directory not found')

        # image_list_adv = os.listdir(FLAGS.input_dir_adv)
        # image_list_adv = [_ for _ in image_list_adv if _.endswith('.png')]

        # if len(image_list_adv)==0:
        #     raise Exception('No png files in the input directory')
        #
        # image_list_adv_temp = sorted(image_list_adv)
        # image_list_adv = [os.path.join(FLAGS.input_dir_adv, _) for _ in image_list_adv_temp]
        # image_list_nat = [os.path.join(FLAGS.input_dir_nat, _) for _ in image_list_adv_temp]

        image_list_adv,image_list_nat = get_image(FLAGS.input_dir_adv, FLAGS.input_dir_nat, 30)
        print('image_list_adv=',len(image_list_adv))
        image_list_adv_tensor = tf.convert_to_tensor(image_list_adv, dtype=tf.string)
        image_list_nat_tensor = tf.convert_to_tensor(image_list_nat, dtype=tf.string)

        with tf.variable_scope('load_image'):
            # define the image list queue
            # image_list_adv_queue = tf.train.string_input_producer(image_list_adv, shuffle=False, capacity=FLAGS.name_queue_capacity)
            # image_list_nat_queue = tf.train.string_input_producer(image_list_nat, shuffle=False, capacity=FLAGS.name_queue_capacity)
            #print('[Queue] image list queue use shuffle: %s'%(FLAGS.mode == 'Train'))
            output = tf.train.slice_input_producer([image_list_adv_tensor, image_list_nat_tensor],
                                                   shuffle=False, capacity=FLAGS.name_queue_capacity)

            # Reading and decode the images
            reader = tf.WholeFileReader(name='image_reader')
            image_adv = tf.read_file(output[0])
            image_nat = tf.read_file(output[1])
            input_image_adv = tf.image.decode_png(image_adv, channels=3)
            input_image_nat = tf.image.decode_png(image_nat, channels=3)
            input_image_adv = tf.image.convert_image_dtype(input_image_adv, dtype=tf.float32)
            input_image_nat = tf.image.convert_image_dtype(input_image_nat, dtype=tf.float32)

            assertion = tf.assert_equal(tf.shape(input_image_adv)[2], 3, message="image does not have 3 channels")
            with tf.control_dependencies([assertion]):
                input_image_adv = tf.identity(input_image_adv)
                input_image_nat = tf.identity(input_image_nat)

            # Normalize the low resolution image to [0, 1],
            a_image = preprocessadv(input_image_adv)
            # b_image = preprocessadv(input_image_nat)
            # high resolution to [-1, 1]
            b_image = preprocess(input_image_nat)

            inputs, targets = [a_image, b_image]

        # The data augmentation part
        with tf.name_scope('data_preprocessing'):
            with tf.name_scope('random_crop'):
                # Check whether perform crop
                if (FLAGS.random_crop is True) and FLAGS.mode == 'train':
                    print('[Config] Use random crop')
                    # Set the shape of the input image. the target will have 4X size
                    input_size = tf.shape(inputs)
                    target_size = tf.shape(targets)
                    # tf.floor() xiangxia quzheng
                    offset_w = tf.cast(tf.floor(tf.random_uniform([], 0, tf.cast(input_size[1], tf.float32) - FLAGS.crop_size)),
                                       dtype=tf.int32)
                    offset_h = tf.cast(tf.floor(tf.random_uniform([], 0, tf.cast(input_size[0], tf.float32) - FLAGS.crop_size)),
                                       dtype=tf.int32)

                    
                    if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
                        inputs = tf.image.crop_to_bounding_box(inputs, offset_h, offset_w, FLAGS.crop_size,
                                                               FLAGS.crop_size)
                        targets = tf.image.crop_to_bounding_box(targets, offset_h, offset_w, FLAGS.crop_size,
                                                                FLAGS.crop_size)
                        # targets = tf.image.crop_to_bounding_box(targets, offset_h*2, offset_w*2, FLAGS.crop_size*2,
                        #                                         FLAGS.crop_size*2)
                    elif FLAGS.task == 'denoise':
                        inputs = tf.image.crop_to_bounding_box(inputs, offset_h, offset_w, FLAGS.crop_size,
                                                               FLAGS.crop_size)
                        targets = tf.image.crop_to_bounding_box(targets, offset_h, offset_w,
                                                                FLAGS.crop_size, FLAGS.crop_size)
                # Do not perform crop
                else:
                    inputs = tf.identity(inputs)
                    targets = tf.identity(targets)

            with tf.variable_scope('random_flip'):
                # Check for random flip:
                if (FLAGS.flip is True) and (FLAGS.mode == 'train'):
                    print('[Config] Use random flip')
                    # Produce the decision of random flip
                    decision = tf.random_uniform([], 0, 1, dtype=tf.float32)

                    input_images = random_flip(inputs, decision)
                    target_images = random_flip(targets, decision)
                else:
                    input_images = tf.identity(inputs)
                    target_images = tf.identity(targets)

            if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
                input_images.set_shape([FLAGS.crop_size, FLAGS.crop_size, 3])
                target_images.set_shape([FLAGS.crop_size, FLAGS.crop_size, 3])
                # target_images.set_shape([FLAGS.crop_size*2, FLAGS.crop_size*2, 3])
            elif FLAGS.task == 'denoise':
                input_images.set_shape([FLAGS.crop_size, FLAGS.crop_size, 3])
                target_images.set_shape([FLAGS.crop_size, FLAGS.crop_size, 3])

        if FLAGS.mode == 'train':
            paths_adv_batch, paths_nat_batch, inputs_batch, targets_batch = tf.train.shuffle_batch([output[0], output[1], input_images, target_images],
                                            batch_size=FLAGS.batch_size, capacity=FLAGS.image_queue_capacity+4*FLAGS.batch_size,
                                            min_after_dequeue=FLAGS.image_queue_capacity, num_tnateads=FLAGS.queue_thread)
        else:
            paths_adv_batch, paths_nat_batch, inputs_batch, targets_batch = tf.train.batch([output[0], output[1], input_images, target_images],
                                            batch_size=FLAGS.batch_size, num_threads=FLAGS.queue_thread, allow_smaller_final_batch=True)

        steps_per_epoch = int(math.ceil(len(image_list_adv) / FLAGS.batch_size))
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
            inputs_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])
            targets_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])
            # targets_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size*2, FLAGS.crop_size*2, 3])
        elif FLAGS.task == 'denoise':
            inputs_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])
            targets_batch.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])
    return Data(
        paths_adv=paths_adv_batch,
        paths_nat=paths_nat_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        image_count=len(image_list_adv),
        steps_per_epoch=steps_per_epoch
    )


# The test data loader. Allow input image with different size
def test_data_loader(FLAGS):
    # Get the image name list
    if (FLAGS.input_dir_adv == 'None') or (FLAGS.input_dir_nat == 'None'):
        raise ValueError('Input directory is not provided')

    if (not os.path.exists(FLAGS.input_dir_adv)) or (not os.path.exists(FLAGS.input_dir_nat)):
        raise ValueError('Input directory not found')

    image_list_adv_temp = os.listdir(FLAGS.input_dir_adv)
    image_list_adv = [os.path.join(FLAGS.input_dir_adv, _) for _ in image_list_adv_temp if _.split('.')[-1] == 'png']
    image_list_nat = [os.path.join(FLAGS.input_dir_nat, _) for _ in image_list_adv_temp if _.split('.')[-1] == 'png']

    # Read in and preprocess the images
    def preprocess_test(name, mode):
        im = sic.imread(name, mode="RGB").astype(np.float32)
        # check grayscale image
        if im.shape[-1] != 3:
            h, w = im.shape
            temp = np.empty((h, w, 3), dtype=np.uint8)
            temp[:, :, :] = im[:, :, np.newaxis]
            im = temp.copy()
        if mode == 'adv':
            im = im / np.max(im)
        elif mode == 'nat':
            im = im / np.max(im)
            im = im * 2 - 1

        return im

    image_adv = [preprocess_test(_, 'adv') for _ in image_list_adv]
    image_nat = [preprocess_test(_, 'nat') for _ in image_list_nat]

    # Push path and image into a list
    Data = collections.namedtuple('Data', 'paths_adv, paths_nat, inputs, targets')

    return Data(
        paths_adv = image_list_adv,
        paths_nat = image_list_nat,
        inputs = image_adv,
        targets = image_nat
    )


# The inference data loader. Allow input image with different size
def inference_data_loader(FLAGS):
    # Get the image name list
    if (FLAGS.input_dir_adv == 'None'):
        raise ValueError('Input directory is not provided')

    if not os.path.exists(FLAGS.input_dir_adv):
        raise ValueError('Input directory not found')

    image_list_adv_temp = os.listdir(FLAGS.input_dir_adv)
    image_list_adv = [os.path.join(FLAGS.input_dir_adv, _) for _ in image_list_adv_temp if _.split('.')[-1] == 'png']
    # image_list_adv = get_test_image(FLAGS.input_dir_adv,'IncV3')
    # image_list_adv = get_test_image(FLAGS.input_dir_adv,'IncResV2')
    # image_list_adv = get_test_image(FLAGS.input_dir_adv,'ResV2')

    # sec_dirs_list = [i[0] for i in os.walk(FLAGS.input_dir_adv)][1:]
    # image_list_adv = []
    # for sec_dir_path in sec_dirs_list:
    #     for filename in os.listdir(sec_dir_path):
    #         image_list_adv.append(os.path.join(sec_dir_path,filename))
    # print(len(image_list_adv))
    # Read in and preprocess the images
    def preprocess_test(name):
        im = sic.imread(name, mode="RGB").astype(np.float32)
        # im = sic.imresize(im,(256,256))
        # added by zsd
        # h, w, _ = im.shape
        # im = im[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)

        # check grayscale image
        if im.shape[-1] != 3:
            h, w = im.shape
            temp = np.empty((h, w, 3), dtype=np.uint8)

            temp[:, :, :] = im[:, :, np.newaxis]

            im = temp.copy()
        im = im / np.max(im)

        return im

    image_adv = [preprocess_test(_) for _ in image_list_adv]

    # Push path and image into a list
    Data = collections.namedtuple('Data', 'paths_adv, inputs')

    return Data(
        paths_adv=image_list_adv,
        inputs=image_adv
    )

'''added by zsd'''

def evaluaute(FLAGS,gen_output,val_data):
    pass
    print('Im here')
    # In the testing time, no flip and crop is needed


        # Declare the test data reader
    fr = open('/home/lthpc/workspace/zhangshudong/adve/test_adve/inception_v3/1000label.txt', 'r')
    dic = {}
    dic1 = {}
    keys = []
    x = 0
    for line in fr:
        a = line.strip().split(':')
        # print(a)  # a[0] represent picture name      a[1] represent picture label
        # dic[a[0]] = a[1]  # image name -> str label
        dic1[a[0]] = x  # hang -> str label
        x += 1
        keys.append(a[0])
    fr.close()


    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
    # path_adv = tf.placeholder(tf.string, shape=[], name='path_adv')

    # g2=tf.Graph()
    tf.reset_default_graph()

    print('Finish building the network')
    checkpoint_path = FLAGS.inception_ckpt




    # init = tf.global_variables_initializer()
    gen_output = preprocess_image(gen_output,299,299,is_training=False)
    gen_output=tf.expand_dims(gen_output,axis=0)
    print(gen_output.shape)
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(gen_output, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
        prediction = tf.argmax(logits,1)

    variables_to_restore = slim.get_variables_to_restore()
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore)





    # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
    # var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    # weight_initiallizer = tf.train.Saver(var_list)

    # Define the initialization operation
    # init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_fn(sess)
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        # weight_initiallizer.restore(sess, FLAGS.checkpoint)

        max_iter = len(val_data.inputs)
        print('Evaluation starts!!')
        count = 0
        for i in range(max_iter):
            input_im = np.array([val_data.inputs[i]]).astype(np.float32)

            path_adv = val_data.paths_adv[i]
            label=dic1[path_adv.split('_')[0].split('/')[-1]]
            prediction_ = sess.run(prediction, feed_dict={inputs_raw: input_im, path_adv: path_adv})
            print(prediction_)
            if prediction_ == label:

                count+=1
        print("Accuracy=",count/max_iter)

# Definition of the generator
def generator(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise  ValueError('No FLAGS is provided for generator')

    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
            net = batchnorm(net, FLAGS.is_training)
            net = prelu_tf(net)
            net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
            net = batchnorm(net, FLAGS.is_training)
            net = net + inputs

        return net


    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, FLAGS.num_resblock+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv2(net, 3, 64, 1, use_bias=False, scope='conv')
            net = batchnorm(net, FLAGS.is_training)

        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            net = conv2(net, 3, 256, 1, scope='conv')
            # fangda liangbei
            net = pixelShuffler(net, scale=1)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = conv2(net, 3, 256, 1, scope='conv')
            
            net = pixelShuffler(net, scale=1)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = conv2(net, 9, gen_output_channels, 1, scope='conv')

    return net


# Definition of the discriminator
def discriminator(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            net = batchnorm(net, FLAGS.is_training)
            net = lrelu(net, 0.2)

        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit'):
            # The input layer
            with tf.variable_scope('input_stage'):
                net = conv2(dis_inputs, 3, 64, 1, scope='conv')
                net = lrelu(net, 0.2)

            # The discriminator block part
            # block 1
            net = discriminator_block(net, 64, 3, 2, 'disblock_1')

            # block 2
            net = discriminator_block(net, 128, 3, 1, 'disblock_2')

            # block 3
            net = discriminator_block(net, 128, 3, 2, 'disblock_3')

            # block 4
            net = discriminator_block(net, 256, 3, 1, 'disblock_4')

            # block 5
            net = discriminator_block(net, 256, 3, 2, 'disblock_5')

            # block 6
            net = discriminator_block(net, 512, 3, 1, 'disblock_6')

            # block_7
            net = discriminator_block(net, 512, 3, 2, 'disblock_7')

            # The dense layer 1
            with tf.variable_scope('dense_layer_1'):
                net = slim.flatten(net)
                net = denselayer(net, 1024)
                net = lrelu(net, 0.2)

            # The dense layer 2
            with tf.variable_scope('dense_layer_2'):
                net = denselayer(net, 1)
                net = tf.nn.sigmoid(net)

    return net


def VGG19_slim(input, type, reuse,scope):
    # Define the feature to extract according to the type of perceptual
    if type == 'VGG54':
        target_layer =  'vgg_19/conv5/conv5_4'
    elif type == 'VGG34':
        target_layer =  'vgg_19/conv3/conv3_4'
    else:
        raise NotImplementedError('Unknown perceptual type')
    _, output = vgg_19(input, is_training=False, reuse=reuse)

    output = output[target_layer]

    return output

def IncV3_slim(input,type,reuse,scope):

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, output = inception.inception_v3(input, num_classes=1, is_training=False, reuse=reuse)

        output = output[type]
    # if type == "Mixed_7c":
    #     out_split = tf.split(output,4,axis=3)
    #     output_1 = out_split[1]
    #
    # elif type == "Mixed_5d":
    #     out_split = tf.split(output, 4, axis=3)
    #     output_1 = out_split[1]
    #     print("hahahahaha")
    #     print(output_1.shape)
    # else:
    #     raise NotImplementedError('Unknown perceptual type')



    return output


# Define the whole network architecture
def Resnet(inputs, targets, FLAGS):
    # Define the container of the parameter
    Network = collections.namedtuple('Network', 'content_loss, gen_grads_and_vars, gen_output, train, global_step, \
            learning_rate')

    # Build the generator part
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)
        # gen_output.set_shape([FLAGS.batch_size, FLAGS.crop_size * 2, FLAGS.crop_size * 2, 3])
        gen_output.set_shape([FLAGS.batch_size, FLAGS.crop_size , FLAGS.crop_size , 3])

    # Use the VGG54 feature
    if FLAGS.perceptual_mode == 'VGG54':
        with tf.name_scope('vgg19_1') as scope:
            extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

    elif FLAGS.perceptual_mode == 'VGG34':
        with tf.name_scope('vgg19_1') as scope:
            extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

        # Use inception feature
    elif FLAGS.perceptual_mode == 'Mixed_7c':
        with tf.name_scope('IncV3_1') as scope:
            extracted_feature_gen = IncV3_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
        with tf.name_scope('IncV3_2') as scope:
            extracted_feature_target = IncV3_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

    elif FLAGS.perceptual_mode == 'Mixed_5d':
        with tf.name_scope('IncV3_1') as scope:
            extracted_feature_gen = IncV3_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
        with tf.name_scope('IncV3_2') as scope:
            extracted_feature_target = IncV3_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

    elif FLAGS.perceptual_mode == 'MSE':
        extracted_feature_gen = gen_output
        extracted_feature_target = targets

    else:
        raise NotImplementedError('Unknown perceptual type')

    # Calculating the generator loss
    with tf.variable_scope('generator_loss'):
        # Content loss
        with tf.variable_scope('content_loss'):
            # Compute the euclidean distance between the two features
            # check=tf.equal(extracted_feature_gen, extracted_feature_target)
            diff = extracted_feature_gen - extracted_feature_target
            if FLAGS.perceptual_mode == 'MSE':
                content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
            else:
                content_loss = FLAGS.vgg_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

        gen_loss = content_loss

    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                                   staircase=FLAGS.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('generator_train'):
        # Need to wait discriminator to perform train step
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    # [ToDo] If we do not use moving average on loss??
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([content_loss])

    return Network(
        content_loss=exp_averager.average(content_loss),
        gen_grads_and_vars=gen_grads_and_vars,
        gen_output=gen_output,
        train=tf.group(update_loss, incr_global_step, gen_train),
        global_step=global_step,
        learning_rate=learning_rate)



def save_images(fetches, FLAGS, step=None):
    # image_dir = os.path.join(FLAGS.output_dir, "images_checkpoint_feature125000_fgsm")
    image_dir = FLAGS.output_dir
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    in_path = fetches['path_adv']
    name, _ = os.path.splitext(os.path.basename(str(in_path)))
    fileset = {"name": name, "step": step}

    if FLAGS.mode == 'inference':
        kind = "outputs"
        filename = name + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][0]
        with open(out_path, "wb") as f:
            f.write(contents)
        filesets.append(fileset)
    else:
        # for kind in ["inputs", "outputs", "targets"]:
        for kind in [ "outputs"]:
            filename = name + ".png"
            # filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][0]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets












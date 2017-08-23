import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load model and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # TODO: sess.graph instead of tf.get_default_graph()?
    # graph = tf.get_default_graph()
    graph = sess.graph

    # Tensorboard
    # tf.summary.FileWriter('runs', graph)

    # Get tensors from graph
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # TODO: conv2d or conv2d_transpose?
    # https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/595f35e6
    # -b940-400f-afb2-2015319aa640/lessons/69fe4a9c-656e-46c8-bc32-aee9e60b8984/concepts/3dcaf318-9e4b-4bb6-b057
    # -886c254abd44

    # vgg_layer7_out size (?, 1, 1, 4096)
    # vgg_layer4_out size (?, 10, 36, 512)
    # vgg_layer3_out size (?, 20, 72, 256)

    # TODO: 1x1 convolution on layer 7
    # 1x1 convolution
    # kernel_size = 1
    # strides = (1, 1)
    # vgg_layer7_out = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size, strides=strides)

    # Transpose
    kernel_size = 2
    strides = (2, 2)
    filters = 512
    output = tf.layers.conv2d_transpose(vgg_layer7_out, filters, kernel_size, strides=strides)
    # print(output.get_shape())
    # Size is (?, 10, 36, 512)

    # Add skip layer
    output = tf.add(output, vgg_layer4_out)
    # print(output.get_shape())
    # Size is (?, 10, 36, 512)

    # Transpose
    kernel_size = 2
    strides = (2, 2)
    filters = 256
    output = tf.layers.conv2d_transpose(output, filters, kernel_size, strides=strides)
    # print(output.get_shape())
    # Size is (?, 20, 72, 256)

    # Add skip layer
    output = tf.add(output, vgg_layer3_out)
    # print(output.get_shape())
    # Size is (?, 20, 72, 256)

    # Transpose
    kernel_size = 8
    strides = (8, 8)
    filters = num_classes
    output = tf.layers.conv2d_transpose(output, filters, kernel_size, strides=strides)
    # print(output.get_shape())
    # Size is (?, 160, 576, num_classes)

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # TODO: Verify train_op, loss and optimizer
    # TODO: Shape of correct_label?
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())

    num_train_images = 289
    num_batches = num_train_images // batch_size

    # TODO: Implement function
    print('*** Starting training ***')
    for i in range(epochs):
        print('*** Epoch {} ***'.format(i+1))

        # Get generator
        batch_generator = get_batches_fn(batch_size)

        # Get one batch of images
        for i in range(num_batches):
            images, gt_images = next(batch_generator)

            # Feed dictionary
            feed_dict = {
                input_image: images,
                correct_label: gt_images,
                keep_prob: 1.0,
                learning_rate: 0.001
            }

            # Run operations
            sess.run(train_op, feed_dict=feed_dict)

            if i == num_batches-1 or i % 10 == 0:
                loss = sess.run(cross_entropy_loss, feed_dict=feed_dict)
                print("Cross Entropy Loss: ", loss)

# Test batch function looks strange
#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # (5, 18) in final layer
    data_dir = './data'
    runs_dir = './runs'
    model_dir = 'vgg'
    training_dir = 'data_road/training'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Hyperparameters
    epochs = 20
    batch_size = 4

    correct_label = tf.placeholder(tf.float32, name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, model_dir)

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, training_dir), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        pre_training = time.time()

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                 input_image, correct_label, keep_prob, learning_rate)

        post_training = time.time()

        print('')
        print('*** Training Comlete ***')
        print('Duration: %.0f seconds' % (post_training - pre_training))
        print('')

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

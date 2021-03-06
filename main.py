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
    graph = sess.graph

    # Uncomment to view graph in Tensorboard
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

    # 1x1 convolutions can be used to add layers with num_classes channels instead of 256/512.
    # It was found that this technique resultet in worse performance.
    """
    kernel_size = 1
    strides = (1, 1)
    filters = num_classes
    layer7_fcn = tf.layers.conv2d(vgg_layer7_out, filters, kernel_size, strides=strides, padding='same')
    layer4_fcn = tf.layers.conv2d(vgg_layer4_out, filters, kernel_size, strides=strides, padding='same')
    layer3_fcn = tf.layers.conv2d(vgg_layer3_out, filters, kernel_size, strides=strides, padding='same')
    """

    # Transpose convolution
    kernel_size = 2
    strides = (2, 2)
    filters = 512
    output = tf.layers.conv2d_transpose(vgg_layer7_out, filters, kernel_size,strides=strides, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # Size is (?, 10, 36, 512)

    # Skip layer
    output = tf.add(output, vgg_layer4_out)
    # Size is (?, 10, 36, 512)

    # Transpose convolution
    kernel_size = 2
    strides = (2, 2)
    filters = 256
    output = tf.layers.conv2d_transpose(output, filters, kernel_size, strides=strides, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # Size is (?, 20, 72, 256)

    # Skip layer
    output = tf.add(output, vgg_layer3_out)
    # Size is (?, 20, 72, 256)

    # Transpose convolution
    kernel_size = 8
    strides = (8, 8)
    filters = num_classes
    output = tf.layers.conv2d_transpose(output, filters, kernel_size, strides=strides, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
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

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, merged_summary=None, filewriter=None):
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
    :param merged_summary: Scalar summaries for Tensorboard
    :param filewriter: FileWriter for Tensorboard
    """
    # Hyperparameters
    dropout_keep_prob = 0.8
    start_learning_rate = 0.001
    learning_rate_decay = 0.98

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Calculate number of batches
    num_train_images = 289
    num_batches = num_train_images // batch_size

    # Number of trained images for Tensorboard graph
    image_count = 0

    # Batches between logging
    num_batches_per_log = 25

    print('*** Starting training ***')
    for i in range(epochs):
        print('*** Epoch {} ***'.format(i+1))

        # Get generator
        batch_generator = get_batches_fn(batch_size)

        # Get one batch of images
        for batch, (images, gt_images) in enumerate(batch_generator):

            # Feed dictionary
            feed_dict = {
                input_image: images,
                correct_label: gt_images,
                keep_prob: dropout_keep_prob,
                learning_rate: start_learning_rate * learning_rate_decay ** i
            }

            # Run train operation
            sess.run(train_op, feed_dict=feed_dict)

            # Print and log cross entropy loss
            if batch == num_batches-1 or (batch+1) % num_batches_per_log == 0:
                feed_dict[keep_prob] = 1.0

                # Save loss to disk for Tensorboard visualization
                try:
                    summary = sess.run(merged_summary, feed_dict=feed_dict)
                    filewriter.add_summary(summary, image_count)
                    image_count += batch_size * num_batches_per_log
                except (TypeError, AttributeError):
                    if i == 0 and batch == 0:
                        print('Not logging to Tensorboard')

                loss = sess.run(cross_entropy_loss, feed_dict=feed_dict)
                print("Cross Entropy Loss: ", loss)
tests.test_train_nn(train_nn)


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
    epochs = 30
    batch_size = 8

    # Save log and inference samples
    training_name = 'k10b08e30do08lrd098-kernel-init'
    save_dir = os.path.join(runs_dir, 'train', training_name)

    # Placeholders
    correct_label = tf.placeholder(tf.float32, name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, model_dir)

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, training_dir), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Get layers from pretrained VGG graph
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # Get last layer from fully convolutional network using vgg layers, skip layers, and transpose convolutions
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # Get logits, Tensorflow train and loss operations
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Log Cross Entropy for visualization in Tensorboard
        tf.summary.scalar('cross_entropy', cross_entropy_loss)
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(save_dir)

        # Start timer
        pre_training = time.time()

        # Train
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                 input_image, correct_label, keep_prob, learning_rate,
                 merged_summary=merged_summary, filewriter=train_writer)

        # End timer
        post_training = time.time()

        print('')
        print('*** Training Complete ***')
        print('Duration: %.0f seconds' % (post_training - pre_training))
        print('')

        # Save inference samples if running longer training
        if epochs >= 20:
            helper.save_inference_samples(save_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

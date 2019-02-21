
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


learning_rate = 0.001
num_steps = 300000
batch_size = 10
size = 64
display_step = 50
examples_to_show = 10


num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 32 # 2nd layer num features (the latent dim)
num_input = 500*500 # MNIST data input (img shape: 28*28)
num_output = 10

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 500 ,500,1])
Y = tf.placeholder("float", [None, num_output])


npypathx = r'D:\RockGan\3DGANServer\tool\image_1.db.npy'
npypathy = r'D:\RockGan\3DGANServer\tool\label_1.db.npy'
npypathtest =  r'D:\RockGan\3DGANServer\tool\test_6.db.npy'

x_load = np.load(npypathx)
x_load = x_load.reshape([-1,500,500,1])
y_load = np.load(npypathy)
y_load = y_load.reshape([-1,2])
test_load = np.load(npypathtest)
test_load = test_load.reshape([-1,500,500,1])




# Building the encoder
def conv_net(x,reuse):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet',reuse=reuse):
        x = tf.reshape(x, [-1, 500, 500, 1])
        conv1 = tf.layers.conv2d(x, 32, 5,strides=(3, 3), activation=tf.nn.relu)
        conv1 = tf.layers.batch_normalization(conv1,training=True)

        conv2 = tf.layers.conv2d(conv1, 64, 5,strides=(3, 3), activation=tf.nn.relu)
        conv2 = tf.layers.batch_normalization(conv2,training=True)

        conv3 = tf.layers.conv2d(conv2, 128, 5,strides=(3, 3), activation=tf.nn.relu)
        conv3 = tf.layers.batch_normalization(conv3,training=True)

        conv4 = tf.layers.conv2d(conv3, 128, 5,strides=(3, 3), activation=tf.nn.relu)
        conv4 = tf.layers.batch_normalization(conv4,training=True)

        fc1 = tf.contrib.layers.flatten(conv4)
        fc1 = tf.layers.dense(fc1, 1024)

        out = tf.layers.dense(fc1,10)

    return out
# Construct model
encoder_op = conv_net(X,reuse = False)
test_op = conv_net(X,reuse = True)
# Prediction
y_pred = encoder_op
# Targets (Labels) are the input data.
y_true = Y

# Define loss and optimizer, minimize the squared error
#loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=encoder_op, labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#Start Training
#Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):

        idx = np.random.randint(len(x_load), size=batch_size)
        batch_x = x_load[idx]/255
        batch_y = y_load[idx]
        tmpy0 = (batch_y[0:,0]+ 90 )  / 18
        tmpy1 = (batch_y[0:, 1] + 30) / 40

        tmps = np.array([],dtype=float)
        for number in range(0,batch_size):
            cou   = int(tmpy0[number])
            tmp = np.zeros([10])
            tmp[cou] = 1

            tmps = np.append(tmps, tmp)
            #tmp = np.append(tmp, tmpy1[number])
        batch_y = np.reshape(tmps,[batch_size,10])
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x,Y:batch_y})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

        if i % 200 ==  0 :
            batch_test = test_load/255
            acc = sess.run([test_op], feed_dict={X: batch_test})
            id =np.where(acc == np.max(acc))[2][0]
            print(acc)
            print("number = %i" %  id )
            print("Angle = %i " % (id*18-85))
        # if i % 100000 == 0:
        #     if not os.path.exists(model_directory):
        #         os.makedirs(model_directory)
        #     saver.save(sess, save_path=model_directory + '/Larry_4_' + str(i) + '.cptk')
        #     Y=[]
        #     for i in range(0,199):
        #         batch_x = x_load[i]
        #         batch_x = batch_x.reshape([1,64*64])
        #         g = sess.run(encoder_op, feed_dict={X: batch_x})
        #
        #         Y.append(g)
        #     Y = np.array(Y)
        #     np.save('Y.npy', Y)

        # if i == 100001:
        #     y= np.load('Y.npy')
        #     n = 4
        #     canvas_orig = np.empty((size * n, size * n))
        #     canvas_recon = np.empty((size * n, size * n))
        #     for i in range(n):
        #         # MNIST test set
        #         # batch_x, _ = mnist.test.next_batch(n)
        #         idx = np.random.randint(len(y), size=n)
        #         print(idx)
        #         batch_x = y[idx]
        #         batch_x= batch_x.reshape([-1,32])
        #         # Encode and decode the digit image
        #         g = sess.run(decoder_op2, feed_dict={Z: batch_x})
        #
        #         # Display original images
        #         # for j in range(n):
        #         #     # Draw the original digits
        #         #     canvas_orig[i * size:(i + 1) * size, j * size:(j + 1) * size] = \
        #         #         batch_x[j].reshape([size, size])
        #         # Display reconstructed images
        #         for j in range(n):
        #             # Draw the reconstructed digits
        #             canvas_recon[i * size:(i + 1) * size, j * size:(j + 1) * size] = \
        #                 g[j].reshape([size, size])
        #
        #     print("Original Images")
        #     plt.figure(figsize=(n, n))
        #     plt.imshow(canvas_orig, origin="upper", cmap="gray")
        #     plt.show()
        #
        #     print("Reconstructed Images")
        #     plt.figure(figsize=(n, n))
        #     plt.imshow(canvas_recon, origin="upper", cmap="gray")
        #     plt.show()






        # if i % 3000 == 0 :
        #     num = i
        #     n = 4
        #     canvas_orig = np.empty((size * n, size * n))
        #     canvas_recon = np.empty((size * n, size * n))
        #     for i in range(n):
        #         # MNIST test set
        #         # batch_x, _ = mnist.test.next_batch(n)
        #         if num < 3000:
        #             idx = np.random.randint(len(x_load), size=n)
        #             batch_x = x_load[idx]
        #         else:
        #             idx = np.random.randint(len(x_test), size=n)
        #             batch_x = x_test[idx]
        #         # Encode and decode the digit image
        #         g = sess.run(decoder_op, feed_dict={X: batch_x})
        #
        #         # Display original images
        #         for j in range(n):
        #             # Draw the original digits
        #             canvas_orig[i * size:(i + 1) * size, j * size:(j + 1) * size] = \
        #                 batch_x[j].reshape([size, size])
        #         # Display reconstructed images
        #         for j in range(n):
        #             # Draw the reconstructed digits
        #             canvas_recon[i * size:(i + 1) * size, j * size:(j + 1) * size] = \
        #                 g[j].reshape([size, size])
        #
        #     print("Original Images")
        #     plt.figure(figsize=(n, n))
        #     plt.imshow(canvas_orig, origin="upper", cmap="gray")
        #     plt.show()
        #
        #     print("Reconstructed Images")
        #     plt.figure(figsize=(n, n))
        #     plt.imshow(canvas_recon, origin="upper", cmap="gray")
        #     plt.show()
    # Testing
    # Encode and decode images from test set and visualize their reconstruction.



# def ganGetOneResult(trained_model_path):
#     with tf.Session() as sess:
#         sess.run(init)
#         saver.restore(sess, trained_model_path)
#         Y=[]
#         for i in range(0,99):
#             batch_x = x_load[i]
#             batch_x = batch_x.reshape([1,64*64])
#             g = sess.run(encoder_op, feed_dict={X: batch_x})
#             Y.append(g)
#         Y = np.array(Y)
#         np.save('Ytest_201_300.npy', Y)
#
# ganGetOneResult(trained_model_path)
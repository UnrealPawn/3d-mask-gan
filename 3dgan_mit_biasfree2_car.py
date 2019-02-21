#!/usr/bin/env python

from PIL import Image
import numpy as np
import visdom

import dataIO as d
from utils import *
import glob
from skimage import measure
import cv2,os,csv
import numpy as np
from random import choice
list = []
from PIL import ImageGrab
'''
Global Parameters
'''
n_epochs = 30000
batch_size = 1

g_lr = 0.00025
d_lr = 0.00025

beta = 0.5
d_thresh = 0.8
z_size = 128
y_dim = 8
leak_value = 0.2
cube_len = 64
obj_ratio = 0.7
obj = 'stone'
ysize = 32
train_sample_directory = './train_sample/'
model_directory = './models/'
npypath = r'./Image_new.db.npy'

is_local = False


# config = tf.ConfigProto(
#     device_count={'GPU': 0}
# )
weights = {}
mask_load = np.load(r"./mask_new.db.npy")
mask_load  =mask_load.reshape([-1,64,64,64,1])

z_load = np.load(npypath)
z_load = z_load.reshape([-1,64,64,1])

# one_array = np.zeros((64, 64), float)
# for i in range(64):
#     for j in range(64):
#         if z_load[9][i][j] == True:
#             one_array[i][j] = 255
#         else:
#             one_array[i][j] = 0
#
# one_array = Image.fromarray(one_array)
#
# if one_array.mode != 'RGB':
#     one_array = one_array.convert('RGB')
# one_array.save('123.png')

# y_load = np.load(npypathy)
# y_load = y_load.reshape([-1,ysize])
def generator(z, y, batch_size=batch_size, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]

    with tf.variable_scope("gen", reuse=reuse):
        # z = tf.reshape(z, (batch_size, 1, 1, 1, z_size)) batch, long, width ,height, in_channels

        g_00 =  tf.contrib.layers.conv2d(z, num_outputs=32, kernel_size=3,
                           stride=2, activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm, padding='SAME',
                           weights_initializer=tf.random_normal_initializer(0, 0.02))
        g_01 = tf.contrib.layers.conv2d(g_00, num_outputs=64, kernel_size=3,
                                        stride=2, activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
                                        padding='SAME',
                                        weights_initializer=tf.random_normal_initializer(0, 0.02))

        z = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(g_01), 128, activation_fn=lrelu,
                                              weights_initializer=tf.random_normal_initializer(0, 0.02))

        z = tf.reshape(z, (batch_size, 1, 1, 1, 128))
        yb = tf.reshape(y, shape=[batch_size, 1, 1, 1, y_dim])
        z = tf.concat([z, yb], 4)
        g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size, 4, 4, 4, 512), strides=[1, 1, 1, 1, 1],
                                     padding="VALID")
        g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase_train)

        g_1 = tf.nn.relu(g_1)

        g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size, 8, 8, 8, 256), strides=strides, padding="SAME")
        g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
        g_2 = tf.nn.relu(g_2)

        g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size, 16, 16, 16, 128), strides=strides,
                                     padding="SAME")
        g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
        g_3 = tf.nn.relu(g_3)

        g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size, 32, 32, 32, 64), strides=strides, padding="SAME")
        g_4 = tf.contrib.layers.batch_norm(g_4, is_training=phase_train)
        g_4 = tf.nn.relu(g_4)

        g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'], (batch_size, 64, 64, 64, 1), strides=strides, padding="SAME")

        g_5 = tf.nn.tanh(g_5)#Larry

    print(g_1, 'g1')
    print(g_2, 'g2')
    print(g_3, 'g3')
    print(g_4, 'g4')
    print(g_5, 'g5')

    return g_5


def discriminator(inputs, y, phase_train=True, reuse=False):
    strides = [1, 2, 2, 2, 1]
    with tf.variable_scope("dis", reuse=reuse):
        print("input shape:")
        print(inputs.shape)
        print("mask shape：")
        print(y.shape)
        inputs = tf.concat([inputs, y], 4)
        d_1 = tf.nn.conv3d(inputs, weights['wd1'], strides=strides, padding="SAME")
        d_1 = tf.contrib.layers.batch_norm(d_1, is_training=phase_train)
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, weights['wd2'], strides=strides, padding="SAME")
        d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        d_2 = lrelu(d_2, leak_value)

        d_3 = tf.nn.conv3d(d_2, weights['wd3'], strides=strides, padding="SAME")
        d_3 = tf.contrib.layers.batch_norm(d_3, is_training=phase_train)
        d_3 = lrelu(d_3, leak_value)

        d_4 = tf.nn.conv3d(d_3, weights['wd4'], strides=strides, padding="SAME")
        d_4 = tf.contrib.layers.batch_norm(d_4, is_training=phase_train)
        d_4 = lrelu(d_4)

        d_5 = tf.nn.conv3d(d_4, weights['wd5'], strides=[1, 1, 1, 1, 1], padding="VALID")

        d_5_no_sigmoid = d_5
        d_5 = tf.nn.sigmoid(d_5)

    print(d_1, 'd1')
    print(d_2, 'd2')
    print(d_3, 'd3')
    print(d_4, 'd4')
    print(d_5, 'd5')

    return d_5, d_5_no_sigmoid


def initialiseWeights():
    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()

    # weights['wcov1'] = tf.get_variable("wcov1", shape=[2, 2, 1, 8], initializer=xavier_init)  # Larry
    # weights['wcov2'] = tf.get_variable("wcov2", shape=[2, 2, 8, 32], initializer=xavier_init)  # Larry

    weights['encoder_h1'] = tf.Variable(tf.random_normal([1024, 512]))
    weights['encoder_h2'] = tf.Variable(tf.random_normal([512, 128]))
    weights['encoder_b1'] = tf.Variable(tf.random_normal([512]))
    weights['encoder_b2'] = tf.Variable(tf.random_normal([128]))

    weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 512, 128 + y_dim], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wg5'] = tf.get_variable("wg5", shape=[4, 4, 4, 1, 64], initializer=xavier_init)

    weights['wd1'] = tf.get_variable("wd1", shape=[4, 4, 4, 2, 64], initializer=xavier_init)
    weights['wd2'] = tf.get_variable("wd2", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wd3'] = tf.get_variable("wd3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wd4'] = tf.get_variable("wd4", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
    weights['wd5'] = tf.get_variable("wd5", shape=[4, 4, 4, 512, 1], initializer=xavier_init)

    return weights


def trainGAN(is_dummy=False, checkpoint=None):
    weights = initialiseWeights()

    z_vector = tf.placeholder(shape=[batch_size, z_size,z_size,1], dtype=tf.float32)
    #z_vector = tf.placeholder(shape=[batch_size, 32,32,1], dtype=tf.float32)
    y_vector = tf.placeholder(dtype = tf.float32, shape=[batch_size, y_dim])
    x_vector = tf.placeholder(shape=[batch_size, cube_len, cube_len, cube_len, 1], dtype=tf.float32)

    net_g_train = generator(z_vector, y_vector, phase_train=True, reuse=False)

    y_mask = tf.placeholder(tf.float32, shape=[batch_size, 64, 64, 64, 1])

    d_output_x, d_no_sigmoid_output_x = discriminator(x_vector, y_mask, phase_train=True, reuse=False)
    d_output_x = tf.maximum(tf.minimum(d_output_x, 0.99), 0.01)
    summary_d_x_hist = tf.summary.histogram("d_prob_x", d_output_x)

    d_output_z, d_no_sigmoid_output_z = discriminator(net_g_train, y_mask, phase_train=True, reuse=True)
    d_output_z = tf.maximum(tf.minimum(d_output_z, 0.99), 0.01)
    summary_d_z_hist = tf.summary.histogram("d_prob_z", d_output_z)

    # Compute the discriminator accuracy
    n_p_x = tf.reduce_sum(tf.cast(d_output_x > 0.5, tf.int32))
    n_p_z = tf.reduce_sum(tf.cast(d_output_z < 0.5, tf.int32))
    d_acc = tf.divide(n_p_x + n_p_z, 2 * batch_size)


    d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_no_sigmoid_output_x, labels=tf.ones_like(
        d_output_x)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=d_no_sigmoid_output_z,
                                                               labels=tf.zeros_like(d_output_z))
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_no_sigmoid_output_z, labels=tf.ones_like(d_output_z))

    d_loss = tf.reduce_mean(d_loss)
    gen_loss_L1 = tf.reduce_mean(tf.abs(x_vector - net_g_train))
    g_loss = tf.reduce_mean(g_loss)*0.1 + gen_loss_L1*0.9

    summary_d_loss = tf.summary.scalar("d_loss", d_loss)
    summary_g_loss = tf.summary.scalar("g_loss", g_loss)
    summary_n_p_z = tf.summary.scalar("n_p_z", n_p_z)
    summary_n_p_x = tf.summary.scalar("n_p_x", n_p_x)
    summary_d_acc = tf.summary.scalar("d_acc", d_acc)



    para_g = [var for var in tf.trainable_variables() if
              any(x in var.name for x in ['encoder_h', 'encoder_b', 'wg', 'bg', 'gen'])]
    para_d = [var for var in tf.trainable_variables() if
              any(x in var.name for x in ['encoder_h', 'encoder_b', 'wd', 'bd', 'dis'])]

    # only update the weights for the discriminator network
    optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr, beta1=beta).minimize(d_loss, var_list=para_d)
    optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=beta).minimize(g_loss, var_list=para_g)

    saver = tf.train.Saver()
    # vis = visdom.Visdom()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())


        if is_dummy:
            volumes = np.random.randint(0, 2, (batch_size, cube_len, cube_len, cube_len))
            print('Using Dummy Data')
        else:
            volumes = d.getAll(obj=obj, train=True, is_local=is_local, obj_ratio=obj_ratio)
            print('Using ' + obj + ' Data')
        volumes = volumes[..., np.newaxis].astype(np.float)

        for epoch in range(n_epochs):

            idx = np.random.randint(len(volumes), size=batch_size)
            x = volumes[idx]
            y = np.random.normal(0, 0.33, size=[batch_size, y_dim]).astype(np.float32)
            z = z_load[idx]

            y_mask_array = mask_load[idx]
            d_summary_merge = tf.summary.merge([summary_d_loss,
                                                summary_d_x_hist,
                                                summary_d_z_hist,
                                                summary_n_p_x,
                                                summary_n_p_z,
                                                summary_d_acc])

            summary_d, discriminator_loss = sess.run([d_summary_merge, d_loss],
                                                     feed_dict={z_vector: z, x_vector: x, y_vector: y,
                                                                y_mask: y_mask_array})
            summary_g, generator_loss = sess.run([summary_g_loss, g_loss],
                                                 feed_dict={z_vector: z, x_vector: x, y_vector: y,
                                                            y_mask: y_mask_array})
            d_accuracy, n_x, n_z = sess.run([d_acc, n_p_x, n_p_z],
                                            feed_dict={z_vector: z, x_vector: x, y_vector: y, y_mask: y_mask_array})
            print(n_x, n_z)

            if d_accuracy < d_thresh:
                sess.run([optimizer_op_d], feed_dict={z_vector: z, x_vector: x, y_vector: y, y_mask: y_mask_array})
                # write_Excle.saveValue('D', epoch, discriminator_loss.item(), generator_loss.item(), d_accuracy)
                print('Discriminator Training ', "epoch: ", epoch, ', d_loss:', discriminator_loss, 'g_loss:',
                      generator_loss, "d_acc: ", d_accuracy)

                list.append(
                    dict(netType='D', epoch=epoch, d_loss=discriminator_loss.item(), g_loss=generator_loss.item(),
                         d_acc=d_accuracy))
            if d_accuracy > 0.5:
                sess.run([optimizer_op_g], feed_dict={z_vector: z, x_vector: x, y_vector: y, y_mask: y_mask_array})
                # write_Excle.saveValue('G', epoch, discriminator_loss.item(), generator_loss.item(), d_accuracy)

                list.append(
                    dict(netType='G', epoch=epoch, d_loss=discriminator_loss.item(), g_loss=generator_loss.item(),
                         d_acc=d_accuracy))

            print('Generator Training ', "epoch: ", epoch, ', d_loss:', discriminator_loss, 'g_loss:', generator_loss,
                  "d_acc: ", d_accuracy)


            if epoch % 400 == 10:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, save_path=model_directory + '/Larry_2_' + str(epoch) + '.cptk')


IsInit = False
Istrue = False
number = 0
min_x = min_y = min_z = max_x = max_y = max_z = 0


def SetBoundState(ix, iy, iz, ax, ay, az):
    global min_x, min_y, min_z, max_x, max_y, max_z
    min_x = ix
    min_y = iy
    min_z = iz
    max_x = ax
    max_y = ay
    max_z = az


def ganGetOneResult(trained_model_path):
    global IsInit
    global Istrue
    global number

    if IsInit == False:
        initialiseWeights()
        IsInit = True


    z_vector = tf.placeholder(shape=[batch_size, z_size, z_size, 1], dtype=tf.float32)
    y_vector = tf.placeholder(tf.float32, [batch_size, y_dim])
    net_g_test = generator(z_vector, y_vector, phase_train=True)
    saver = tf.train.Saver()
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, trained_model_path)
            mynumber = 0
            flag = False
            if flag:
                dir = []
                accsum = 0
                volumes = d.getAll(obj=obj, train=True, is_local=is_local, obj_ratio=obj_ratio)
                volumes = volumes.reshape((-1, 64, 64, 64))
                for n1 in range(0,len(volumes)):
                    data_path = r"D:/Paper\image\test\A/" + str(n1+900) +"y.png"
                    img = Image.open(data_path)
                    img.save('output.png', quality=100)

                    def matrixpic(img):
                        matrixpic = np.asarray(img)
                        (h, w) = matrixpic.shape[:2]
                        for i in range(h):
                            for j in range(w):
                                if matrixpic[i][j] == 255:
                                    matrixpic[i][j] = 255
                                else:
                                    matrixpic[i][j] = 0
                        return matrixpic

                    # 绿色转白色
                    def convertcolor(img):
                        img = np.asarray(img)
                        (h, w) = img.shape[:2]
                        for i in range(h):
                            for j in range(w):
                                if img[i][j][1] > 180 and img[i][j][0] < 20 and img[i][j][2] < 20:
                                    img[i][j] = 255
                        return img

                    # 生成二值测试图像
                    img = cv2.imread('output.png')
                    img1 = np.ones(img.shape[:2])
                    converted = convertcolor(img)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    thresh = matrixpic(gray)

                    #
                    # 检测所有图形的轮廓
                    contours = measure.find_contours(thresh, 0.5)
                    x = 0
                    y = 0
                    for n, contour in enumerate(contours):
                        if contour.size > x:
                            x = contour.size
                            y = contour
                    y[:, [0, 1]] = y[:, [1, 0]]
                    _contours = []
                    _contours.append(np.around(np.expand_dims(y, 1)).astype(np.int))
                    cv2.drawContours(img1, _contours, -1, (0, 0, 0), -1)

                    img1 = cv2.resize(img1, (64, 64), interpolation=cv2.INTER_CUBIC)

                    img_np_1d = img1 * 255

                    one_array = np.zeros((64, 64), float)
                    for i in range(64):
                        for j in range(64):
                            if img_np_1d[i][j] < 155:
                                one_array[i][j] = 255
                            else:
                                one_array[i][j] = 0

                    one_array = Image.fromarray(one_array)
                    one_array = one_array.rotate(-90)
                    if one_array.mode != 'RGB':
                        one_array = one_array.convert('RGB')
                    one_array.save('1234.png')
                    one_array = cv2.imread('1234.png', 0)
                    left = up = down = right = 0
                    for j in range(32):
                        for i in range(64):
                            if one_array[i][j] != 0:
                                left = j
                        if left != 0:
                            break

                    for i in range(32):
                        for j in range(64):
                            if one_array[i][j] == 255:
                                up = i
                        if up != 0:
                            break
                    for i in range(32):
                        for j in range(64):
                            if one_array[63 - i][j] == 255:
                                down = 63 - i
                        if down != 0:
                            break
                    right = down - up + 24
                    box = (left, up, right, down)
                    img = Image.open('1234.png')
                    roi = img.crop(box)
                    dst = roi.resize((64, 64), Image.ANTIALIAS)
                    dst.save('123456.png', quality=100)
                    one_array = cv2.imread('123456.png', 0) / 255
                    Istrue = True
                    x = []
                    for i in range(0, 5):
                        x.append(one_array)
                    x = np.array(x)
                    z_sample = np.array(x)
                    z_sample = z_sample.reshape([-1, z_size, z_size, 1])
                    y = np.random.normal(0, 0.33, size=[batch_size, y_dim]).astype(np.float32)
                    g_objects = sess.run(net_g_test, feed_dict={z_vector: z_sample, y_vector: y})
                    id_ch = np.random.randint(0, batch_size, 4)
                    i = 0
                    if g_objects[id_ch[i]].max() > 0.5:
                        result = np.squeeze(g_objects[id_ch[i]] > 0.5)


                        g_objects = volumes[n1]
                        x = []
                        for i in range(0, 5):
                            x.append(g_objects)
                        g_objects = np.array(x)
                        id_ch = np.random.randint(0, batch_size, 4)
                        i = 0
                        result2 = []
                        if g_objects[id_ch[i]].max() > 0.5:
                            result2 = np.squeeze(g_objects[id_ch[i]] > 0.5)
                        pre1 = np.sum(np.logical_xor(result, result2))
                        pre2 = np.sum(np.logical_and(result, result2))
                        pre3 = np.sum(np.logical_or(result, result2))
                        print("xor=", pre1, "   and=", pre2, "   or=", pre3)
                        acc = pre2 / pre3
                        print("Acc=", acc)
                        accsum = accsum+acc
                        dir.append([n1,acc])

                dir.append(["sum",len(volumes), accsum])
                with open('acc.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    for row in dir:
                        writer.writerow(row)
                f.close()

            while True:
                if Istrue:
                    x = []
                    Istrue = False
                    volumes = d.getAll(obj=obj, train=True, is_local=is_local, obj_ratio=obj_ratio)

                    volumes=volumes.reshape((-1, 64, 64, 64))
                    imagepath = r"D:/im/" + str(mynumber - 1 + 900) + ".png"
                    im = ImageGrab.grab()
                    im.save(imagepath)
                    g_objects =volumes[num]


                    for i in range(0, 5):
                        x.append(g_objects)
                    g_objects = np.array(x)
                    id_ch = np.random.randint(0, batch_size, 4)
                    i = 0
                    print(g_objects[id_ch[i]].max(), g_objects[id_ch[i]].min(), g_objects[id_ch[i]].shape)
                    if g_objects[id_ch[i]].max() > 0.5:
                        result = np.squeeze(g_objects[id_ch[i]] > 0.5)
                        yield result
                else:

                    data_path =glob.glob( r'D:\Paper\image\test\A\*.png')
                    onepath = choice(data_path)
                    print(onepath)

                    onepath = r"D:/Paper\image\test\A/" + str(mynumber + 900) + "y.png"
                    imagepath = r"D:/im/" + str(mynumber-1 + 900) +"y.png"
                    im = ImageGrab.grab()
                    im.save(imagepath)
                    print(onepath)
                    mynumber = mynumber + 1
                    img = Image.open(onepath)
                    img.save('output.png', quality=100)
                    def matrixpic(img):
                        matrixpic = np.asarray(img)
                        (h, w) = matrixpic.shape[:2]
                        for i in range(h):
                            for j in range(w):
                                if matrixpic[i][j] == 255:
                                    matrixpic[i][j] = 255
                                else:
                                    matrixpic[i][j] = 0
                        return matrixpic

                    # 绿色转白色
                    def convertcolor(img):
                        img = np.asarray(img)
                        (h, w) = img.shape[:2]
                        for i in range(h):
                            for j in range(w):
                                if img[i][j][1] > 180 and img[i][j][0] < 20 and img[i][j][2] < 20:
                                    img[i][j] = 255
                        return img

                    # 生成二值测试图像
                    img = cv2.imread('output.png')
                    img1 = np.ones(img.shape[:2])
                    converted = convertcolor(img)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    thresh = matrixpic(gray)

                    #
                    # 检测所有图形的轮廓
                    contours = measure.find_contours(thresh, 0.5)
                    x = 0
                    y = 0
                    for n, contour in enumerate(contours):
                        if contour.size > x:
                            x = contour.size
                            y = contour
                    y[:, [0, 1]] = y[:, [1, 0]]
                    _contours = []
                    _contours.append(np.around(np.expand_dims(y, 1)).astype(np.int))
                    cv2.drawContours(img1, _contours, -1, (0, 0, 0), -1)

                    img1 = cv2.resize(img1, (64, 64), interpolation=cv2.INTER_CUBIC)


                    img_np_1d = img1 * 255

                    one_array = np.zeros((64, 64), float)
                    for i in range(64):
                        for j in range(64):
                            if img_np_1d[i][j] < 155:
                                one_array[i][j] = 255
                            else:
                                one_array[i][j] = 0

                    one_array = Image.fromarray(one_array)
                    one_array = one_array.rotate(-90)
                    if one_array.mode != 'RGB':
                        one_array = one_array.convert('RGB')
                    one_array.save('1234.png')
                    one_array = cv2.imread('1234.png', 0)
                    left = up = down = right = 0
                    for j in range(32):
                        for i  in range(64):
                            if one_array[i][j] != 0:
                                left = j
                        if left != 0:
                            break

                    for i in range(32):
                        for j in range(64):
                            if one_array[i][j] == 255:
                                up = i
                        if up != 0:
                            break
                    for i in range(32):
                        for j in range(64):
                            if one_array[63 - i][j] == 255:
                                down = 63 - i
                        if down != 0:
                            break
                    right = down - up + 24
                    box = (left, up, right, down)
                    img = Image.open('1234.png')
                    roi = img.crop(box)
                    dst = roi.resize((64, 64), Image.ANTIALIAS)
                    dst.save('123456.png', quality=100)
                    num = onepath[-8:-5]
                    num = int(num) - 901
                   #
                    one_array = cv2.imread('123456.png', 0)
                    for i in range(64):
                        for j in range(64):
                            if one_array[i][j] > 155:
                                one_array[i][j] = 1
                            else:
                                one_array[i][j] = 0
                    Istrue = True
                    x = []
                    for i in range(0, 5):
                        x.append(one_array)
                    x = np.array(x)
                    z_sample = np.array(x)
                    z_sample = z_sample.reshape([-1, z_size, z_size, 1])
                    y = np.random.normal(0, 0.33, size=[batch_size, y_dim]).astype(np.float32)
                    g_objects = sess.run(net_g_test, feed_dict={z_vector: z_sample, y_vector: y})
                    id_ch = np.random.randint(0, batch_size, 4)
                    i = 0
                    print(g_objects[id_ch[i]].max(), g_objects[id_ch[i]].min(), g_objects[id_ch[i]].shape)
                    if g_objects[id_ch[i]].max() > 0.5:
                        result = np.squeeze(g_objects[id_ch[i]] > 0.5)

                        # volumes = d.getAll(obj=obj, train=True, is_local=is_local, obj_ratio=obj_ratio)
                        # volumes = volumes.reshape((-1, 64, 64, 64))
                        # g_objects = volumes[num]
                        # x = []
                        # for i in range(0, 5):
                        #     x.append(g_objects)
                        # g_objects = np.array(x)
                        # id_ch = np.random.randint(0, batch_size, 4)
                        # i = 0
                        # print(g_objects[id_ch[i]].max(), g_objects[id_ch[i]].min(), g_objects[id_ch[i]].shape)
                        # result2 = []
                        # result3 = []
                        # result4 = []
                        # if g_objects[id_ch[i]].max() > 0.5:
                        #     result2 = np.squeeze(g_objects[id_ch[i]] > 0.5)
                        #
                        # pre1 = np.sum(np.logical_xor(result, result2))
                        # pre2 = np.sum(np.logical_and(result, result2))
                        # pre3 = np.sum(np.logical_or(result, result2))
                        # print("xor=", pre1, "   and=", pre2, "   or=", pre3)
                        # acc = pre2 / pre3
                        # print("Acc=", acc)

                        yield result
    yield None


if __name__ == '__main__':
    trainGAN(is_dummy=False, checkpoint=None)

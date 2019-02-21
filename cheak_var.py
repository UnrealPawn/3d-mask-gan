import tensorflow as tf

import numpy as np
model_directory_1 = './models' + '/Larry_1' + str(16410) + '.cptk'
model_directory_2 = './models' + '/Larry_1' + str(10810) + '.cptk'
reader_1 = tf.train.NewCheckpointReader(model_directory_1)
reader_2 = tf.train.NewCheckpointReader(model_directory_2)

#all_variables = reader_1.get_variable_to_shape_map()

quantized_conv_list = ['gen/Conv/weights', 'gen/Conv_1/weights', 'gen/Conv_2/weights', 'gen/Conv_3/weights']

pf = open('result.txt', 'w+')

for quantized_conv_name in quantized_conv_list:


    weight_1 = reader_1.get_tensor(quantized_conv_name)
    weight_2 = reader_2.get_tensor(quantized_conv_name)
    weight = abs(weight_1- weight_2)
    avr = weight.mean()
    print(quantized_conv_name)

    print('***************************************')

    print(weight.shape)
    print(avr)

    [n, cout, h, w] = weight.shape

    print(cout, h, w)

    pf.write(quantized_conv_name)

    pf.write('\n')

    pf.write(str(n) + ' ' + str(cout) + ' ' + str(h) + ' ' + str(w) + '\n')

    # for c in range(cout):

    # pf.write('***********'+str(c)+'**********\n')

    for n1 in range(n):

        pf.write('***********' + str(n1) + '**********\n')

        for h1 in range(h):

            for w1 in range(w):

                for c in range(cout):
                    pf.write('%f ' % weight[n1][c][h1][w1])

        pf.write('\n')

        # pf.write('\n')

    # try:
    #
    #     bias = reader.get_tensor(quantized_conv_name + "/b")
    #
    #     n2 = bias.shape
    #
    #     print(bias.shape)
    #
    #     print
    #     n2
    #
    #     print
    #     '***************************************'
    #
    #     pf.write('\n')
    #
    #     pf.write('**************************bias:')
    #
    #     pf.write('\n')
    #
    #     pf.write(str(n) + '\n')
    #
    #     # for n1 in range(n2):
    #
    #     #    pf.write('%f, ' %bias[n1])
    #
    #     # pf.write('\n')
    #
    #     for b in bias:
    #         pf.write('%f ' % b)
    #
    # except:
    #
    #     print
    #     'no bias'

    pf.write('\n')

pf.close()

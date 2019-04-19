#!/usr/bin/python3.5
import vgg_structure
import tensorflow as tf
import imagedata
import numpy as np

trainlight='/home/xiaoyue/PycharmProjects/vgg16_learn/image/train_light.list'
trainunknown='/home/xiaoyue/PycharmProjects/vgg16_learn/image/train_unknown.list'
testlight='/home/xiaoyue/PycharmProjects/vgg16_learn/image/test_light.txt'
testunknown='/home/xiaoyue/PycharmProjects/vgg16_learn/image/test_unknown.txt'
x_data=np.zeros([98,32,32,3],dtype='float32')
y_data=np.zeros([98,2])
x_test=np.zeros([16,32,32,3],dtype='float32')
y_test=np.zeros([16,2])

f_trainlight=open(trainlight,'r',encoding="utf-8")
i=0
for line in f_trainlight.readlines():
    x_data[i,:,:,:]=imagedata.getImageData(line)
    y_data[i,:]=[1,0]
    i=i+1
f_trainlight.close()
f_trainunknown = open(trainunknown, 'r', encoding="utf-8")
for line in f_trainunknown.readlines():
    x_data[i,:,:,:]=imagedata.getImageData(line)
    y_data[i,:]=[0,1]
f_trainunknown.close()

f_testlight=open(testlight,'r',encoding="utf-8")
i=0
for line in f_testlight.readlines():
    x_test[i,:,:,:]=imagedata.getImageData(line)
    y_test[i,:]=[1,0]
    i=i+1
f_testlight.close()
f_testunknown = open(testunknown, 'r', encoding="utf-8")
for line in f_testunknown.readlines():
    x_test[i,:,:,:]=imagedata.getImageData(line)
    y_test[i,:]=[0,1]
f_testunknown.close()


batch_size = 16
learning_rate = 0.0001
max_steps = 10000
n_cls = 2


def train():
    # x=tf.placeholder(tf.float32,shape=[None,244,244,3],name='input')
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='input')
    y = tf.placeholder(tf.float32, shape=[None, n_cls], name='labels')
    vgg = vgg_structure.vgg16(x)
    vgg.convlayers()
    vgg.fc_layers()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3, labels=y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    accuraacy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(vgg.fc3, 1), tf.argmax(y, 1)), tf.float32))
    # img_batch,label_batch=tf.train.shuffle_batch([images,labels],batch_size=batch_size)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for j in range(max_steps):
            for i in range (6):
                _, loss_val = sess.run([train_step, loss], feed_dict={x: x_data[i*batch_size:(i+1)*batch_size-1,:,:,:], y: y_data[i*batch_size:(i+1)*batch_size-1,:]})
            if j % 10 == 0:
                train_accuraacy = accuraacy.eval(feed_dict={x: x_test, y: y_test})
                print(sess.run(vgg.fc3, feed_dict={x: x_test, y: y_test}))
                print(" Step: [%d]   Loss: %f, training accuracy: %g" % (j, loss_val, train_accuraacy))
            if j % 50==0:
                saver.save(sess,'./model/model.ckpt', global_step=j)
            if (j + 1) == max_steps:
                saver.save(sess, './model/model.ckpt', global_step=j)


if __name__ == '__main__':
    train()





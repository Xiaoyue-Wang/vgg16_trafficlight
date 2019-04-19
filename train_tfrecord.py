#!/usr/bin/python3.5
import vgg_structure
import tensorflow as tf
import pre_data
batch_size=2
learning_rate=0.0001
max_steps=10000
n_cls=2

def train():

    #x=tf.placeholder(tf.float32,shape=[None,244,244,3],name='input')
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='input')
    y=tf.placeholder(tf.float32,shape=[None,n_cls],name='labels')
    vgg=vgg_structure.vgg16(x)
    vgg.convlayers()
    vgg.fc_layers()
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3,labels=y))
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    accuraacy=tf.cast(tf.equal(tf.argmax(vgg.fc3,1),tf.argmax(y,1)),tf.float32)
    #img_batch,label_batch=tf.train.shuffle_batch([images,labels],batch_size=batch_size)

    init=tf.global_variables_initializer()
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(max_steps):
            #batch_x,batch_y=sess.run([img_batch,label_batch])
            #for i in range (batch_size):
            images0,label0= pre_data.read_tfrecord()
            images1,label1 = pre_data.read_tfrecord()
            label0=tf.one_hot(label0,n_cls,1,0)
            label1 = tf.one_hot(label1, n_cls, 1, 0)
            _,loss_val=sess.run([train_step,loss],feed_dict={x:[images0],y:[label0.eval()]})
            if i%10==0:
                images0, label0 = pre_data.read_tfrecord()
                images1, label1 = pre_data.read_tfrecord()
                label0 = tf.one_hot(label0, n_cls, 1, 0)
                label1 = tf.one_hot(label1, n_cls, 1, 0)
                train_accuraacy=accuraacy.eval(feed_dict={x:[images0],y:[label0.eval()]})
                print(sess.run(vgg.fc3,feed_dict={x:[images0],y:[label0.eval()]}))
                print(" Step: [%d]   Loss: %f, training accuracy: %g" %(i,loss_val,train_accuraacy))
            if(i+1)==max_steps:
                saver.save(sess,'./model/model.ckpt',global_step=i)

if __name__=='__main__':
    train()
    




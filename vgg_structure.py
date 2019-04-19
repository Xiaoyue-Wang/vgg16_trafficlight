import tensorflow as tf

class vgg16:

    def __init__(self,img, weights=None,sess=None):
        self.img=img
        self.convlayers()
        self.fc_layers()
        self.probs=tf.nn.softmax(self.fc3)

        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters=[]


        with tf.name_scope('conv1_1') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,3,64],stddev=0.1,dtype=tf.float32),name='weights')
            conv=tf.nn.conv2d(self.img,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[64],dtype=tf.float32),trainable=True,name='biases')
            self.conv1_1=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel,bias]

        with tf.name_scope('conv1_2') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,64,64],stddev=0.1,dtype=tf.float32,name='weighs'))
            conv=tf.nn.conv2d(self.conv1_1,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[64],dtype=tf.float32),trainable=True,name='biases')
            self.conv1_2=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel, bias]
        self.pool1=self.conv1_2
#        self.pool1=tf.nn.max_pool(self.conv1_2,[1,2,2,1],[1,2,2,1],padding='SAME',name='pooling')

        with tf.name_scope('conv2_1') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1,dtype=tf.float32),name='weights')
            conv=tf.nn.conv2d(self.pool1,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[128],dtype=tf.float32),trainable=True,name='biases')
            self.conv2_1=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel, bias]
        with tf.name_scope('conv2_2') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,128,128],stddev=0.1,dtype=tf.float32,name='weighs'))
            conv=tf.nn.conv2d(self.conv2_1,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[128],dtype=tf.float32),trainable=True,name='biases')
            self.conv2_2=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel, bias]

        self.pool2=tf.nn.max_pool(self.conv2_2,[1,2,2,1],[1,2,2,1],padding='SAME',name='pooling')

        with tf.name_scope('conv3_1') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,128,256],stddev=0.1,dtype=tf.float32),name='weights')
            conv=tf.nn.conv2d(self.pool2,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[256],dtype=tf.float32),trainable=True,name='biases')
            self.conv3_1=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel, bias]
        with tf.name_scope('conv3_2') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,256,256],stddev=0.1,dtype=tf.float32,name='weighs'))
            conv=tf.nn.conv2d(self.conv3_1,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[256],dtype=tf.float32),trainable=True,name='biases')
            self.conv3_2=tf.nn.relu(tf.nn.bias_add(conv,bias))
            self.parameters += [kernel, bias]
        with tf.name_scope('conv3_3') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,256,256],stddev=0.1,dtype=tf.float32,name='weighs'))
            conv=tf.nn.conv2d(self.conv3_2,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[256],dtype=tf.float32),trainable=True,name='biases')
            self.conv3_3=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel, bias]

        self.pool3=tf.nn.max_pool(self.conv3_3,[1,2,2,1],[1,2,2,1],padding='SAME',name='pooling')

        with tf.name_scope('conv4_1') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,256,512],stddev=0.1,dtype=tf.float32),name='weights')
            conv=tf.nn.conv2d(self.pool3,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[512],dtype=tf.float32),trainable=True,name='biases')
            self.conv4_1=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel, bias]
        with tf.name_scope('conv4_2') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,512,512],stddev=0.1,dtype=tf.float32,name='weighs'))
            conv=tf.nn.conv2d(self.conv4_1,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[512],dtype=tf.float32),trainable=True,name='biases')
            self.conv4_2=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel, bias]
        with tf.name_scope('conv4_3') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,512,512],stddev=0.1,dtype=tf.float32,name='weighs'))
            conv=tf.nn.conv2d(self.conv4_2,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[512],dtype=tf.float32),trainable=True,name='biases')
            self.conv4_3=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel, bias]
        self.pool4=self.conv4_3
#        self.pool4=tf.nn.max_pool(self.conv4_3,[1,2,2,1],[1,2,2,1],padding='SAME',name='pooling')

        with tf.name_scope('conv5_1') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,512,1024],stddev=0.1,dtype=tf.float32),name='weights')
            conv=tf.nn.conv2d(self.pool4,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant([1],shape=[1024],dtype=tf.float32),trainable=True,name='biases')
            self.conv5_1=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel, bias]
        with tf.name_scope('conv5_2') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,1024,1024],stddev=0.1,dtype=tf.float32,name='weighs'))
            conv=tf.nn.conv2d(self.conv5_1,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[1024],dtype=tf.float32),trainable=True,dtype=tf.float32,name='biases')
            self.conv5_2=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel, bias]
        with tf.name_scope('conv5_3') as scope:
            kernel=tf.Variable(tf.random_normal([3,3,1024,1024],stddev=0.1,dtype=tf.float32,name='weighs'))
            conv=tf.nn.conv2d(self.conv5_2,kernel,[1,1,1,1],padding='SAME')
            bias=tf.Variable(tf.constant(1,shape=[1024],dtype=tf.float32),trainable=True,name='biases')
            self.conv5_3=tf.nn.relu(tf.nn.bias_add(conv,bias),name=scope)
            self.parameters += [kernel, bias]

        self.pool5=tf.nn.max_pool(self.conv5_3,[1,2,2,1],[1,2,2,1],padding='SAME',name='pooling')

    def fc_layers(self):
        with tf.name_scope('fc1') as scope:
            #shape=(self.pool5).get_shape().as_list()
            #fc_dim=shape[1]*shape[2]*shape[3]
            #print(fc_dim)
            fc_dim=49152
            shape=1
            reshaped=tf.reshape(self.pool5,[shape,fc_dim])
            fc_weight=tf.Variable(tf.random_normal([fc_dim,4096],stddev=0.1,dtype=tf.float32),name="weight")
            fc_bias=tf.Variable(tf.constant(0.1,shape=[4096],dtype=tf.float32),name='bias')
            self.fc1=tf.nn.relu((tf.matmul(reshaped,fc_weight)+fc_bias),name=scope)
            self.fc1=tf.nn.dropout(self.fc1,0.5)
            self.parameters += [fc_weight,fc_bias]

        with tf.name_scope('fc2') as scope:
            fc2_weight=tf.Variable(tf.random_normal([4096,4096],stddev=0.1,dtype=tf.float32))
            fc2_bias=tf.Variable(tf.constant(0.1,shape=[4096],dtype=tf.float32),name='bias')
            self.fc2=tf.nn.relu((tf.matmul(self.fc1,fc2_weight)+fc2_bias),name=scope)
            self.fc2=tf.nn.dropout(self.fc2,0.5)
            self.parameters += [fc2_weight, fc2_bias]

        with tf.name_scope('fc3') as scope:
            fc3_weight=tf.Variable(tf.random_normal([4096,2],stddev=0.1,dtype=tf.float32))
            fc3_bias=tf.Variable(tf.constant(0.1,shape=[2],dtype=tf.float32),name='bias')
            self.fc3=tf.nn.relu((tf.matmul(self.fc2,fc3_weight)+fc3_bias),name=scope)
            self.fc3=tf.nn.dropout(self.fc3,0.5)
            self.parameters += [fc3_weight, fc3_bias]




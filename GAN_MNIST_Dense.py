import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('mnist_data', one_hot=True)
epochs = 150
batch_size = 128
z_dim = 32
h_dim = 256
epoch_len = mnist.train.images.shape[0]/batch_size

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def weight(shape,name):
    return tf.Variable(xavier_init(shape),name=name)

def bias(shape,name):
    return tf.Variable(tf.zeros(shape),name=name)

def feed_forward(x,W,b,activation=tf.nn.relu):
    return activation(tf.matmul(x,W) + b)

DW_1 = weight([784,h_dim],"DW_1")
Db_1 = bias([h_dim],"Db_1")

DW_2 = weight([h_dim,1],"DW_2")
Db_2 = bias([1],"Db_2")

d_var = [DW_1,Db_1,DW_2,Db_2]

def discriminator(images):
    h0 = feed_forward(images,DW_1,Db_1)
    h1 = feed_forward(h0,DW_2,Db_2,tf.identity)

    return h1

GW_1 = weight([z_dim,h_dim],"GW_1")
Gb_1 = bias([h_dim],"Gb_1")

GW_2 = weight([h_dim,784],"GW_2")
Gb_2 = bias([784],"Gb_2")

g_var = [GW_1,Gb_1,GW_2,Gb_2]

def generator(latent):
    h0 = feed_forward(latent, GW_1, Gb_1)
    h1 = feed_forward(h0, GW_2, Gb_2,tf.nn.sigmoid)

    return h1

X = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
Z = tf.placeholder(tf.float32,[None,z_dim])

d_real_logits = discriminator(X)
gen_data = generator(Z)
d_fake_logits = discriminator(gen_data)

disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits,labels=tf.ones_like(d_real_logits))) \
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,labels=tf.zeros_like(d_fake_logits)))

gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,labels=tf.ones_like(d_fake_logits)))

d_train = tf.train.AdamOptimizer().minimize(disc_loss,var_list=d_var)
g_train = tf.train.AdamOptimizer().minimize(gen_loss,var_list=g_var)

saver = tf.train.Saver(max_to_keep=1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if tf.train.checkpoint_exists("GAN_log/model"):
    saver.restore(sess,"GAN_log/model")
    sample_len = 10
    z_sample = np.random.uniform(-1., 1., size=[sample_len*sample_len, z_dim])
    image = sess.run(gen_data,feed_dict={Z:z_sample})
    final = np.zeros([28*sample_len,28*sample_len])

    for i in xrange(sample_len):
        for j in xrange(sample_len):
            final[i*28 : (i+1)*28,j*28 : (j+1)*28] = np.reshape(image[sample_len*i + j],[28,28])

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(final, cmap='Greys_r')
    fig.savefig('Transformation.png', bbox_inches='tight')
    plt.show()
else:
    for ep in xrange(epochs):
        for _ in xrange(epoch_len):
            X_ , y_ = mnist.train.next_batch(batch_size)
            z_sample = np.random.uniform(-1., 1., size=[batch_size, z_dim])
            _, D_loss_curr = sess.run([d_train, disc_loss], feed_dict={X: X_, Z: z_sample, y:y_})
            _, G_loss_curr = sess.run([g_train, gen_loss], feed_dict={Z: z_sample, y:y_})
            _ = sess.run(g_train,feed_dict={Z: z_sample, y:y_})
        print "Epoch: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" %(ep,D_loss_curr,G_loss_curr)
        saver.save(sess,"GAN_log/model")

    sample_len = 10
    z_sample = np.random.uniform(-1., 1., size=[sample_len*sample_len, z_dim])
    image = sess.run(gen_data,feed_dict={Z:z_sample})
    final = np.zeros([28*sample_len,28*sample_len])
    
    for i in xrange(sample_len):
        for j in xrange(sample_len):
            final[i*28 : (i+1)*28,j*28 : (j+1)*28] = np.reshape(image[sample_len*i + j],[28,28])

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(final, cmap='Greys_r')
    fig.savefig('Transformation.png', bbox_inches='tight')
    plt.show()
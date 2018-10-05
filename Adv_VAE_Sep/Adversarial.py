from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# config = tf.ConfigProto(device_count = {'GPU': 0})
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5

mnist = input_data.read_data_sets("mnist_data",one_hot=True)
latent_size = 256
batch_size = 128
epochs = 1000
epoch_len = mnist.train.images.shape[0]/batch_size

c_lr = 1e-3
v_lr = 1e-4

c = 0.2

alpha = 10.
target = [1,0,0,0,0,0,0,0,0,0]

regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
initializer = tf.contrib.layers.xavier_initializer()

if not os.path.exists("Classifier_log/"):
    os.makedirs("Classifier_log/")

if not os.path.exists("VAE_log/"):
    os.makedirs("VAE_log/")

if not os.path.exists("Adv_log/"):
    os.makedirs("Adv_log/")

if not os.path.exists("out/"):
    os.makedirs("out/")

if not os.path.exists("Noise_out/"):
    os.makedirs("Noise_out/")

if not os.path.exists("Noise/"):
    os.makedirs("Noise/")

def plot(samples,h,w):
    fig = plt.figure(figsize=(h, w))
    gs = gridspec.GridSpec(h, w)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def encoder(X,y):
    # Encoder network to map input to mean and log of variance of latent variable distribution
    with tf.variable_scope("VAE/Encoder"):
        X = tf.concat(axis=1,values=[X,y])
        h1 = tf.layers.dense(X,1024,activation=tf.nn.relu,kernel_initializer = initializer)
        h2 = tf.layers.dense(h1,512,activation=tf.nn.relu,kernel_initializer = initializer)
        h3 = tf.layers.dense(h2,256,activation=tf.nn.relu,kernel_initializer = initializer)
        h4 = tf.layers.dense(h3,128,activation=tf.nn.relu,kernel_initializer = initializer)
        z_mean = tf.layers.dense(h4,latent_size,kernel_initializer = initializer)
        z_log_sigma = tf.layers.dense(h4,latent_size,kernel_initializer = initializer)

    return z_mean, z_log_sigma

def decoder_image(z,y,reuse=False):
    # decoder_image network to map latent variable to predicted output
    with tf.variable_scope("VAE/Decoder_Image",reuse=reuse):
        z = tf.concat(axis=1,values=[z,y])
        h1 = tf.layers.dense(z,128,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_image_1")
        h2 = tf.layers.dense(h1,256,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_image_2")
        h3 = tf.layers.dense(h2,512,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_image_3")
        h4 = tf.layers.dense(h3,1024,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_image_4")    
        output = tf.layers.dense(h4,784,kernel_initializer = initializer,name="decoder_image_5")
    return output, tf.nn.sigmoid(output)

def decoder_noise(z,y,reuse=False):
    # decoder_image network to map latent variable to predicted output
    with tf.variable_scope("Decoder_Noise",reuse=reuse):
        z = tf.concat(axis=1,values=[z,y])
        h1 = tf.layers.dense(z,128,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_noise_1")
        h2 = tf.layers.dense(h1,256,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_noise_2")
        h3 = tf.layers.dense(h2,512,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_noise_3")
        h4 = tf.layers.dense(h3,1024,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_noise_4")    
        output = tf.layers.dense(h4,784,kernel_initializer = initializer,name="decoder_noise_5")
    return output, tf.nn.tanh(output)

def classifier(X,reuse=False):
    X = tf.reshape(X,[-1,28,28,1])
    with tf.variable_scope("Classifier",initializer=initializer, regularizer=regularizer,reuse=reuse):
        h0 = tf.layers.conv2d(X,64,[3,3],activation=tf.nn.relu)
        h1 = tf.layers.conv2d(h0,64,[3,3],strides=2,activation=tf.nn.relu)

        h2 = tf.layers.conv2d(h1,64,[3,3],activation=tf.nn.relu)
        h3 = tf.layers.conv2d(h2,64,[3,3], strides=2,activation=tf.nn.relu)
        
        h4 = tf.reshape(h3,[-1,4*4*64])
        h5 = tf.layers.dense(h4,10)
    
    return h5

def KL_loss(mu,log_sigma):
    # Compute the KL Divergence loss
    return -0.5 * tf.reduce_sum(-tf.exp(log_sigma) - tf.square(mu) + 1 + log_sigma,axis=-1)

def image_loss_l2(y_,y):
    # Compute the L2 Image loss
    return tf.reduce_sum(tf.square(y_-y),axis=-1)

def image_loss_bce(y_,y):
    # Compute the binary cross entropy loss
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y,labels=y_),axis=-1)

def ce_loss(logits,labels):
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),axis=-1)

X = tf.placeholder(tf.float32,[None,784]) # Input
epsilon = tf.placeholder(tf.float32,[None,latent_size]) # Sample from normal gaussian
y = tf.placeholder(tf.float32,[None,10])

# ------------------------------- Classifier -----------------------------------------------

logits = classifier(X)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(logits,1)),tf.float32))
c_loss = tf.reduce_mean(ce_loss(logits,y))
c_step = tf.train.AdamOptimizer(c_lr).minimize(c_loss,var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Classifier"))

# -------------------------------- VAE -------------------------------------------------------

mu, log_sigma = encoder(X,y) # mean and log variance of latent distribution
z = mu + tf.exp(log_sigma/2.)*epsilon # latent variable
raw_gen,gen = decoder_image(z,y) # Expected output / reconstruction

im_loss_bce = image_loss_bce(X,raw_gen)
kl_loss = KL_loss(mu,log_sigma)

_,noise = decoder_noise(z,y)
noise = noise/alpha

noisy_image = gen+noise
noisy_image = noisy_image/tf.reduce_max(noisy_image)

s_logits = classifier(noisy_image,True)

ent_loss = ce_loss(s_logits,tf.zeros_like(s_logits) + target)
pix_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(noise), logits = noise),axis=-1)
nzero = tf.cast(tf.count_nonzero(noise,1),tf.float32)
im_loss = image_loss_l2(noise,tf.zeros_like(noise))

hinge_loss = tf.maximum(tf.reduce_sum(tf.square(noise),axis=1) - c, tf.zeros([batch_size]))

v_loss = tf.reduce_mean(im_loss_bce+kl_loss)
adv_loss = tf.reduce_mean(ent_loss + hinge_loss) # Sum of both loss averaged over all inputs

adv_step = tf.train.AdamOptimizer(v_lr).minimize(adv_loss,var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Decoder_Noise")) # Train step
v_step = tf.train.AdamOptimizer(v_lr).minimize(v_loss,var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="VAE")) # Train step

_,img = decoder_image(epsilon,y,True)
_,noi = decoder_noise(epsilon,y,True)

noi = noi/alpha
fin = img+noi
fin = fin / tf.reshape(tf.reduce_max(fin,axis=1),(-1,1))
# fin = fin/tf.reduce_max(fin)

pred = tf.argmax(classifier(fin,True),1)

# ------------------------------- Session and Savers ---------------------------------------------------

c_saver = tf.train.Saver(max_to_keep=1,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Classifier"))
v_saver = tf.train.Saver(max_to_keep=1,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="VAE"))
adv_saver = tf.train.Saver(max_to_keep=1,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Decoder_Noise"))

sess = tf.Session(config=config)
tf.global_variables_initializer().run(session=sess)

# --------------------------------------------- Train Classifier --------------------------------

if tf.train.checkpoint_exists("Classifier_log/model"):
    c_saver.restore(sess,"Classifier_log/model")
    acc = 0

    for i in xrange(20):
        X_ = mnist.test.images[i*500:(i+1)*500,:]
        y_ = mnist.test.labels[i*500:(i+1)*500,:]
        acc += sess.run(accuracy,feed_dict={X:X_, y:y_})
    print "Accuracy Test: %.4f" %(acc/20.)
else:
    for ep in xrange(5):
        for _ in xrange(epoch_len):
            X_,y_ = mnist.train.next_batch(batch_size)
            _= sess.run([c_step], feed_dict={X:X_, y:y_})
        c_saver.save(sess,"Classifier_log/model")
        print ep

# ------------------------------------------- Train adversarial VAE ------------------------------------

lab = np.zeros((100,10))
for i in xrange(10):
    lab[i*10:(i+1)*10,i] = 1

if tf.train.checkpoint_exists("VAE_log/model"):
    v_saver.restore(sess,"VAE_log/model")
else:
    for i in xrange(50):
        loss1 = 0.0
        loss2 = 0.0
        for j in xrange(epoch_len):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            eps = np.random.randn(batch_size,latent_size)
            _,cost1 = sess.run([v_step,v_loss],feed_dict={X:batch_xs,y:batch_ys, epsilon:eps})
            loss1 += cost1

        im = sess.run(img,feed_dict={epsilon:np.random.randn(100,latent_size),y:lab})
        print "Epoch: %s \t Loss VAE: %s" %(i,loss1/epoch_len)
        fig = plot(im,10,10)
        fig.savefig("out/{}.png".format(str(i).zfill(3)),bbox_inches="tight")
        plt.close(fig)    
        v_saver.save(sess,"VAE_log/model")

if tf.train.checkpoint_exists("Adv_log/model"):
    adv_saver.restore(sess,"Adv_log/model")

for i in xrange(epochs):
    loss1 = 0.0
    loss2 = 0.0
    for j in xrange(epoch_len):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        eps = np.random.randn(batch_size,latent_size)
        _ = sess.run(v_step,feed_dict={X:batch_xs,y:batch_ys, epsilon:eps})
        cost1,cost2,_,_ = sess.run([v_loss,adv_loss,adv_step,v_step],feed_dict={X:batch_xs, epsilon:eps,y:batch_ys})
        loss1 += cost1
        loss2 += cost2
        # cost1 , _ = sess.run([v_loss,v_step],feed_dict={X:batch_xs, epsilon:eps,y:batch_ys})
        # loss1 += cost1

    # print "Epoch: %s \t Loss VAE: %s" %(i,loss1/epoch_len)
    # im = sess.run(img,feed_dict={epsilon:np.random.randn(100,latent_size),y:lab})
    print "Epoch: %s \t Loss VAE: %s \t Loss Adv: %s" %(i,loss1/epoch_len, loss2/epoch_len)
    images,labels,im,n = sess.run([fin,pred,img,noi],feed_dict={epsilon:np.random.randn(100,latent_size),y:lab})
    print labels
    fig = plot(images,10,10)
    fig.savefig("Noise_out/{}.png".format(str(i).zfill(3)),bbox_inches="tight")
    plt.close(fig)
    fig = plot(im,10,10)
    fig.savefig("out/{}.png".format(str(i).zfill(3)),bbox_inches="tight")
    plt.close(fig)    
    fig = plot(n,10,10)
    fig.savefig("Noise/{}.png".format(str(i).zfill(3)),bbox_inches="tight")
    plt.close(fig)    
    v_saver.save(sess,"VAE_log/model")
    adv_saver.save(sess,"Adv_log/model")

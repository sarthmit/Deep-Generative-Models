from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/", one_hot=True)
latent_size = 2
batch_size = 128
epochs = 25

initializer = tf.contrib.layers.xavier_initializer()

def encoder(X):
	# Encoder network to map input to mean and log of variance of latent variable distribution
	h1 = tf.layers.dense(X,1024,activation=tf.nn.relu,kernel_initializer = initializer)
	h2 = tf.layers.dense(h1,512,activation=tf.nn.relu,kernel_initializer = initializer)
	h3 = tf.layers.dense(h2,256,activation=tf.nn.relu,kernel_initializer = initializer)
	h4 = tf.layers.dense(h3,128,activation=tf.nn.relu,kernel_initializer = initializer)
	z_mean = tf.layers.dense(h4,latent_size,kernel_initializer = initializer)
	z_log_sigma = tf.layers.dense(h4,latent_size,kernel_initializer = initializer)
	return z_mean, z_log_sigma

def decoder(z):
	# Decoder network to map latent variable to predicted output
	h1 = tf.layers.dense(z,128,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_1")
	h2 = tf.layers.dense(h1,256,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_2")
	h3 = tf.layers.dense(h2,512,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_3")
	h4 = tf.layers.dense(h3,1024,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_4")	
	output = tf.layers.dense(h4,784,kernel_initializer = initializer,name="decoder_5")
	return output, tf.nn.sigmoid(output)

def generator(z):
	h1 = tf.layers.dense(z,128,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_1",reuse=True)
	h2 = tf.layers.dense(h1,256,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_2",reuse=True)
	h3 = tf.layers.dense(h2,512,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_3",reuse=True)
	h4 = tf.layers.dense(h3,1024,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_4",reuse=True)	
	output = tf.layers.dense(h4,784,kernel_initializer = initializer,name="decoder_5",reuse=True)
	return output, tf.nn.sigmoid(output)

def KL_loss(mu,log_sigma):
	# Compute the KL Divergence loss
	return -0.5 * tf.reduce_sum(-tf.exp(log_sigma) - tf.square(mu) + 1 + log_sigma,axis=-1)

def image_loss_l2(y_,y):
	# Compute the L2 Image loss
	return tf.reduce_sum(tf.square(y_-y),axis=-1)

def image_loss_bce(y_,y):
	# Compute the binary cross entropy loss
	return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y,labels=y_),axis=-1)

def vae():
	X = tf.placeholder(tf.float32,[None,784]) # Input
	y = tf.placeholder(tf.float32,[None,10]) # Labels
	X_cond = tf.concat(axis=1,values=[X,y]) # Final input
	epsilon = tf.placeholder(tf.float32,[None,latent_size]) # Sample from normal gaussian
	mu, log_sigma = encoder(X_cond) # mean and log variance of latent distribution
	
	z = mu + tf.exp(log_sigma/2.)*epsilon # latent variable
	z_cond = tf.concat(axis=1,values=[z,y])
	y_,_ = decoder(z_cond) # Expected output / reconstruction

	total_loss = tf.reduce_mean(KL_loss(mu,log_sigma) + image_loss_bce(X,y_)) # Sum of both loss averaged over all inputs
	train_step = tf.train.AdamOptimizer().minimize(total_loss) # Train step
	
	eps_cond = tf.concat(axis=1,values=[epsilon,y])
	_,image = generator(eps_cond)

	sess = tf.Session()
	tf.global_variables_initializer().run(session=sess)

	epoch_len = len(mnist.train.images)/batch_size

	for i in xrange(epochs):
		loss = 0.0
		for j in xrange(epoch_len):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			eps = np.random.randn(batch_size,latent_size)
			cost,_ = sess.run([total_loss,train_step],feed_dict={X:batch_xs, y:batch_ys, epsilon:eps})
			loss += cost
		# plt.imshow(np.reshape(batch_xs[0],[28,28]),cmap="Greys_r")
		# plt.show()
		# plt.imshow(np.reshape(out[0],[28,28]),cmap="Greys_r")
		# plt.show()
		print "Epoch: %s \t Loss: %s" %(i,loss/epoch_len)

	x_test = mnist.test.images
	y_test = mnist.test.labels
	x_test_encoded = []
	for i in xrange(10):
		x_test_encoded.append(sess.run(mu,feed_dict={X:x_test[ i*1000 : (i+1) * 1000], y:y_test[ i*1000 : (i+1) * 1000]}))
	x_test_encoded = np.reshape(x_test_encoded,[-1,latent_size])
	plt.figure(figsize=(6, 6))
	plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=np.argmax(y_test,axis=-1))
	plt.colorbar()
	plt.show()

	figure = np.zeros((28 * 20, 28 * 20))
	for i in xrange(20):
		for j in xrange(20):
			z_sample = np.random.randn(1,latent_size)
			x_decoded = sess.run(image,feed_dict={epsilon:z_sample,y:[[0,0,0,0,0,1,0,0,0,0]]})
			figure[ i * 28 : ( i + 1 ) * 28, j * 28 : ( j + 1 ) * 28 ] = np.reshape(x_decoded,[28,28])
	plt.figure(figsize=(6,6))
	plt.imshow(figure, cmap="Greys_r")
	plt.show()

vae()

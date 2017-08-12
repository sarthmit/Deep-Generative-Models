from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/")
latent_size = 2
batch_size = 128
epochs = 75 

initializer = tf.contrib.layers.xavier_initializer()

def encoder(X):
	# Encoder network to map input to mean and log of variance of latent variable distribution
	h1 = tf.layers.conv2d(X,64,kernel_size=3,strides=2,activation=tf.nn.relu,kernel_initializer = initializer,padding='same')
	h2 = tf.layers.conv2d(h1,64,kernel_size=3,strides=2,activation=tf.nn.relu,kernel_initializer = initializer,padding='same')
	h3 = tf.layers.conv2d(h2,64,kernel_size=3,strides=2,activation=tf.nn.relu,kernel_initializer = initializer,padding='same')
	h4 = tf.layers.conv2d(h3,64,kernel_size=3,strides=2,activation=tf.nn.relu,kernel_initializer = initializer,padding='same')
	h5 = tf.layers.conv2d(h4,64,kernel_size=3,strides=2,activation=tf.nn.relu,kernel_initializer = initializer,padding='same')
	h6 = tf.reshape(h5,[-1,64])
	z_mean = tf.layers.dense(h6,latent_size,kernel_initializer = initializer)
	z_log_sigma = tf.layers.dense(h6,latent_size,kernel_initializer = initializer)
	return z_mean, z_log_sigma

def decoder(z):
	# Decoder network to map latent variable to predicted output
	h1 = tf.layers.dense(z,7*7*64,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_1")
	h2 = tf.reshape(h1,[-1,7,7,64])
	h3 = tf.layers.conv2d_transpose(h2,64,8,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_2")
	output = tf.layers.conv2d_transpose(h3,1,15,kernel_initializer = initializer,name="decoder_3")
	return output, tf.nn.sigmoid(output)

def generator(z):
	h1 = tf.layers.dense(z,7*7*64,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_1",reuse=True)
	h2 = tf.reshape(h1,[-1,7,7,64])
	h3 = tf.layers.conv2d_transpose(h2,64,8,activation=tf.nn.relu,kernel_initializer = initializer,name="decoder_2",reuse=True)
	output = tf.layers.conv2d_transpose(h3,1,15,kernel_initializer = initializer,name="decoder_3",reuse=True)
	return output, tf.nn.sigmoid(output)

def KL_loss(mu,log_sigma):
	# Compute the KL Divergence loss
	return -0.5 * tf.reduce_sum(-tf.exp(log_sigma) - tf.square(mu) + 1 + log_sigma,axis=-1)

def image_loss_l2(y_,y):
	# Compute the L2 Image loss
	return tf.reduce_sum(tf.square(y_-y),axis=-1)

def image_loss_bce(X,X_):
	# Compute the binary cross entropy loss
	y_ = tf.reshape(X,[-1,784])
	y = tf.reshape(X_,[-1,784])
	return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y,labels=y_),axis=-1)

def vae():
	X = tf.placeholder(tf.float32,[None,28,28,1]) # Input
	epsilon = tf.placeholder(tf.float32,[None,latent_size]) # Sample from normal gaussian
	mu, log_sigma = encoder(X) # mean and log variance of latent distribution
	z = mu + tf.exp(log_sigma/2.)*epsilon # latent variable
	y,_ = decoder(z) # Expected output / reconstruction
	total_loss = tf.reduce_mean(KL_loss(mu,log_sigma) + image_loss_bce(X,y)) # Sum of both loss averaged over all inputs
	train_step = tf.train.AdamOptimizer().minimize(total_loss) # Train step
	
	_,image = generator(epsilon)

	sess = tf.Session()
	tf.global_variables_initializer().run(session=sess)

	epoch_len = len(mnist.train.images)/batch_size

	for i in xrange(epochs):
		loss = 0.0
		for j in xrange(epoch_len):
			batch_xs = np.reshape(mnist.train.next_batch(batch_size)[0],[batch_size,28,28,1])
			eps = np.random.randn(batch_size,latent_size)
			cost,_,out,lat = sess.run([total_loss,train_step,y,z],feed_dict={X:batch_xs, epsilon:eps})
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
		x_test_encoded.append(sess.run(mu,feed_dict={X:np.reshape(x_test[ i*1000 : (i+1) * 1000],[1000,28,28,1])}))
	x_test_encoded = np.reshape(x_test_encoded,[-1,latent_size])
	plt.figure(figsize=(6, 6))
	plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
	plt.colorbar()
	plt.show()

	figure = np.zeros((28 * 20, 28 * 20))
	for i in xrange(20):
		for j in xrange(20):
			z_sample = np.random.randn(1,latent_size)
			x_decoded = sess.run(image,feed_dict={epsilon:z_sample})
			figure[ i * 28 : ( i + 1 ) * 28, j * 28 : ( j + 1 ) * 28 ] = np.reshape(x_decoded,[28,28])
	plt.figure(figsize=(6,6))
	plt.imshow(figure, cmap="Greys_r")
	plt.show()

	# grid_x = norm.ppf(np.linspace(0.05, 0.95, 10))
	# grid_y = norm.ppf(np.linspace(0.05, 0.95, 10))
	# figure = np.zeros((28 * 10, 28 * 10))
	# for i, yi in enumerate(grid_x):
	#     for j, xi in enumerate(grid_y):
	#         z_sample = np.array([[xi, yi]])
	#         x_decoded = sess.run(image,feed_dict={epsilon:z_sample})
	#         digit = x_decoded[0].reshape(28, 28)
	#         figure[i * 28: (i + 1) * 28,
	#                j * 28: (j + 1) * 28] = digit

	# plt.figure(figsize=(10, 10))
	# plt.imshow(figure, cmap='Greys_r')
	# plt.show()

	# for _ in xrange(2):
	# 	a = np.random.randn(1,latent_size)
	# 	img = sess.run(image,feed_dict={epsilon:a})
	# 	img = np.reshape(img,[28,28])
	# 	plt.imshow(img,cmap='Greys_r')
	# 	plt.show()

vae()

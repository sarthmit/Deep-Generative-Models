<h1> Deep-Generative-Models </h1>

<h2> Variational Autoencoders </h2>

<h3> MNIST_VAE_Dense.py </h3>

Using only dense layers.

<h4> Latent variables dimension = 256: </h4>

Getting the distribution of latent space over any 2 dimensions from 256 does not make much sense and we can't visualize the digits generated over all dimensions of the latent space, so we sample random variables from latent space and generate digits from them.

![](https://user-images.githubusercontent.com/19748754/29238286-bd461412-7f4f-11e7-9845-695b0d09f3ed.png)

<h4> Latent variables dimension = 2: </h4>

Run the program to train it over MNIST dataset and get the latent state structure, randomly generated digits and how the generated digits vary over the latent variables space.

Latent Structure           |  Generated Digits        |  Generated Digits over Latent Space
:-------------------------:|:-------------------------: | :----------------------------------:
![](https://user-images.githubusercontent.com/19748754/29238283-bd44e132-7f4f-11e7-8839-27010784ddf4.png)  |  ![](https://user-images.githubusercontent.com/19748754/29238284-bd44df3e-7f4f-11e7-8d8b-3a5d976e012d.png)  |  <img src="https://user-images.githubusercontent.com/19748754/29238285-bd45d02e-7f4f-11e7-9be8-905df10c256a.png" width= 68% height= 28% />

<h3> MNIST_VAE_Conv.py </h3>

Using Convolution layers and Transposed Convolution layers.

<h4> Latent variables dimension = 256: </h4>

Getting the distribution of latent space over any 2 dimensions from 256 does not make much sense and we can't visualize the digits generated over all dimensions of the latent space, so we sample random variables from latent space and generate digits from them.

![](https://user-images.githubusercontent.com/19748754/29492913-65bff302-85a8-11e7-86fd-f2f0dd558aea.png)

<h4> Latent variables dimension = 2: </h4>

Run the program to train it over MNIST dataset and get the latent state structure and randomly generated digits

Latent Structure           |  Generated Digits        |  Generated Digits over Latent Space
:-------------------------:|:-------------------------: | :----------------------------------:
![](https://user-images.githubusercontent.com/19748754/29492915-65c6a454-85a8-11e7-8b28-1125d9610258.png)  |  ![](https://user-images.githubusercontent.com/19748754/29492914-65c0794e-85a8-11e7-8893-8623229a7cc4.png)  |  <img src="https://user-images.githubusercontent.com/19748754/29493457-fed4d5c0-85b3-11e7-9c60-19dbd4115224.png" width= 68% height= 28% />

<h2> Generative Adversarial Networks </h2>

<h3> GAN_MNIST_Dense.py </h3>

Using only dense layers.

<h4> Latent variables dimension = 32: </h4>

Getting the distribution of latent space over any 2 dimensions from 256 does not make much sense and we can't visualize the digits generated over all dimensions of the latent space, so we sample random variables from latent space and generate digits from them.

![](https://user-images.githubusercontent.com/19748754/29492912-65bdc7b2-85a8-11e7-930d-c934e772dfe9.png)

# Deep-Generative-Models

<h1> MNIST_VAE_Dense.py </h1>

<h3> Latent variables dimension = 256: </h3>

Getting the distribution of latent space over any 2 dimensions from 256 does not make much sense and we can't visualize the digits generated over all dimensions of the latent space, so we sample random variables from latent space and generate digits from them.

![](https://user-images.githubusercontent.com/19748754/29238286-bd461412-7f4f-11e7-9845-695b0d09f3ed.png)

<h3> Latent variables dimension = 2: </h3>

Run the program to train it over MNIST dataset and get the latent state structure, randomly generated digits and how the generated digits vary over the latent variables space.

Latent Structure           |  Generated Digits        |  Generated Digits over Latent Space
:-------------------------:|:-------------------------: | :----------------------------------:
![](https://user-images.githubusercontent.com/19748754/29238283-bd44e132-7f4f-11e7-8839-27010784ddf4.png)  |  ![](https://user-images.githubusercontent.com/19748754/29238284-bd44df3e-7f4f-11e7-8d8b-3a5d976e012d.png)  |  <img src="https://user-images.githubusercontent.com/19748754/29238285-bd45d02e-7f4f-11e7-9be8-905df10c256a.png" width= 68% height= 28% />


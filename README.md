# Adversarial Autoencoders
### [Under development]
<img src="https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/cover.png" alt="Cover" style="width: 100px;"/>

This repository contains code to implement adversarial autoenocders using Tensorflow.

The explanation for which can be found [here]().

## Installing the dependencies
Install virtualenv and creating a new virtual environment:

    pip install virtualenv
    virtualenv -p /usr/bin/python3 aa

 Install dependencies

    pip3 install -r requirements.txt

***Note:***

* *I'd highly recommend using your GPU during training.*
* *`tf.nn.sigmoid_cross_entropy_with_logits` has a `targets` parameter which
has been changed to `labels` for tensorflow version > r0.12.*

## Dataset
The MNIST dataset will be downloaded automatically and will be made available
in `./Data` directory.


## Training!
### Autoencoder:
#### Architecture:

**~Image of the architecture~**

To train a basic autoencoder run:

        python3 autoencoder.py --train True

* This trains an autoencoder and saves the trained model once every epoch
in the `./Results/Autoencoder` directory.

To load the trained model and generate images passing inputs to the decoder run:

        python3 autoencoder.py --train False

### Adversarial Autoencoder:
#### Architecture:

**~Image of the architecture~**

Training:

        python3 adversarial_autoencoder.py --train True

Load model and explore the latent space:

        python3 adversarial_autoencoder.py --train False

Example of adversarial autoencoder output when the encoder is constrained
to have a stddev of 5.

~ Images of the posterior and prior distributions matching ~


![Adversarial_autoencoder](https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/adversarial_autoencoder_2.png)

### Supervised Adversarial Autoencoder:
#### Architecture:

**~Image of the architecture~**

Training:

        python3 supervised_adversarial_autoencoder.py --train True

Load model and explore the latent space:

        python3 supervised_adversarial_autoencoder.py --train False

Example of disentanglement of style and content:

![supervised_aa](https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/supervised_autoencoder_100.png)

### Semi-Supervised Adversarial Autoencoder:
#### Architecture:

**~Image of the architecture~**

Training:

        python3 supervised_adversarial_autoencoder.py --train True

Load model and explore the latent space:

        python3 supervised_adversarial_autoencoder.py --train False

Classification accuarcy for 1000 labeled images:
~Graph for variation of accuracy~
~Cat dist match~
~Gauss dist match~

***Note:***
* Each run generates a required tensorboard files under `./Results/<model>/<time_stamp_and_parameters>/Tensorboard` directory.
* Use `tensorboard --logdir <tensorboard_dir>` to look at loss variations
and distributions of latent code.

## Thank You
Please share this repo if you find it helpful.

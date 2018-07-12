# Adversarial autoencoders
<img src="https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/nw_architecture.png" alt="Cover" style="width: 100px;"/>

This repository contains code to implement adversarial autoencoder using Tensorflow.

Medium posts:

1. [A Wizard's guide to Adversarial Autoencoders: Part 1. Autoencoders?](https://medium.com/towards-data-science/a-wizards-guide-to-adversarial-autoencoders-part-1-autoencoder-d9a5f8795af4)

2. [A Wizard's guide to Adversarial Autoencoders: Part 2. Exploring the latent space with Adversarial Autoencoders.](https://medium.com/towards-data-science/a-wizards-guide-to-adversarial-autoencoders-part-2-exploring-latent-space-with-adversarial-2d53a6f8a4f9)

3. [A Wizard's guide to Adversarial Autoencoders: Part 3. Disentanglement of style and content.](https://medium.com/towards-data-science/a-wizards-guide-to-adversarial-autoencoders-part-3-disentanglement-of-style-and-content-89262973a4d7)

3. [A Wizard's guide to Adversarial Autoencoders: Part 4. Classify MNIST using 1000 labels.](https://medium.com/towards-data-science/a-wizards-guide-to-adversarial-autoencoders-part-4-classify-mnist-using-1000-labels-2ca08071f95)

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

To train a basic autoencoder run:

        python3 autoencoder.py --train True

* This trains an autoencoder and saves the trained model once every epoch
in the `./Results/Autoencoder` directory.

To load the trained model and generate images passing inputs to the decoder run:

        python3 autoencoder.py --train False

### Adversarial Autoencoder:
#### Architecture:

<img src="https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/AAE%20Block%20Diagram.png" alt="Cover" style="width: 100px;"/>

Training:

        python3 adversarial_autoencoder.py --train True

Load model and explore the latent space:

        python3 adversarial_autoencoder.py --train False

Example of adversarial autoencoder output when the encoder is constrained
to have a stddev of 5.

<img src="https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/AAE%20dist%20match.png" alt="Cover" style="width: 100px;"/>

**_Matching prior and posterior distributions._**


![Adversarial_autoencoder](https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/adversarial_autoencoder_2.png)
**_Distribution of digits in the latent space._**

### Supervised Adversarial Autoencoder:
#### Architecture:

<img src="https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/Supervised%20AAE.png" alt="Cover" style="width: 100px;"/>

Training:

        python3 supervised_adversarial_autoencoder.py --train True

Load model and explore the latent space:

        python3 supervised_adversarial_autoencoder.py --train False

Example of disentanglement of style and content:
<img src="https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/disentanglement%20of%20style%20and%20content.png" alt="Cover" style="width: 100px;"/>

### Semi-Supervised Adversarial Autoencoder:
#### Architecture:
<img src="https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/semi_AAE%20architecture.png" alt="Cover" style="width: 100px;"/>

Training:

        python3 semi_supervised_adversarial_autoencoder.py --train True

Load model and explore the latent space:

        python3 semi_supervised_adversarial_autoencoder.py --train False

Classification accuracy for 1000 labeled images:

<img src="https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/semi_aae_accuracy_with_NN.png" alt="Cover" style="width: 100px;"/>

<img src="https://raw.githubusercontent.com/Naresh1318/Adversarial_Autoencoder/master/README/cat_n_gauss_dist_real_obtained.png" alt="Cover" style="width: 100px;"/>


***Note:***
* Each run generates a required tensorboard files under `./Results/<model>/<time_stamp_and_parameters>/Tensorboard` directory.
* Use `tensorboard --logdir <tensorboard_dir>` to look at loss variations
and distributions of latent code.
* Windows gives an error when `:` is used during folder naming (this is produced during the folder creation for each run).I 
would suggest you to remove the time stamp from `folder_name` variable in the `form_results()` function. Or, just dual boot linux!


## Thank You
Please share this repo if you find it helpful.

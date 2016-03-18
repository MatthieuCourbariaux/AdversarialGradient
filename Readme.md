# AdversarialGradient

## Motivations

This code reproduces some of the experimental results reported in: 
[Improving back-propagation by adding an adversarial gradient](http://arxiv.org/abs/1510.04189).
The paper introduces a very simple variant of 
[adversarial training](http://arxiv.org/abs/1412.6572)
which yields very impressive results on MNIST,
that is to say **about 0.80% error rate with a 2 x 400 ReLU MLP**.

## Requirements

* Python 2.7, Numpy, Scipy
* [Theano](http://deeplearning.net/software/theano/install.html)
* [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html)

## How-to-run-it

Firstly, download the MNIST dataset:

    wget http://deeplearning.net/data/mnist/mnist.pkl.gz
    
Then, run the training script (which contains all the relevant hyperparameters):

    python mnist.py

The training only lasts **5 minutes** on a TitanX GPU.
The best validation error rate should be about **0.83%**,
and the associated test error rate about **0.93%**.

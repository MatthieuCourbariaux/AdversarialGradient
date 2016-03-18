
import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

import theano
import theano.tensor as T

import lasagne

import cPickle
import gzip

import custom
from collections import OrderedDict

if __name__ == "__main__":
    
    batch_size = 32
    print("batch_size = "+str(batch_size))
    num_units = 400
    print("num_units = "+str(num_units))
    num_epochs = 150
    print("num_epochs = "+str(num_epochs))
    activation = T.nnet.relu
    print("activation = T.nnet.relu")
    
    # Decaying LR 
    LR_start = 0.0001
    print("LR_start = "+str(LR_start))
    LR_fin = LR_start
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    
    save_path = "mnist_parameters.npz"
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading MNIST dataset...')
    
    # Loading the MNIST test set
    # You can get mnist.pkl.gz at http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    # bc01 format    
    train_set_X = train_set[0]
    valid_set_X = valid_set[0]
    test_set_X = test_set[0]
    
    # flatten targets
    train_set_t = np.hstack(train_set[1])
    valid_set_t = np.hstack(valid_set[1])
    test_set_t = np.hstack(test_set[1])
    
    # Onehot the targets
    train_set_t = np.float32(np.eye(10)[train_set_t])    
    valid_set_t = np.float32(np.eye(10)[valid_set_t])
    test_set_t = np.float32(np.eye(10)[test_set_t])

    print('Specifying the graph...') 
    
    # Prepare Theano variables for inputs and targets
    X = T.matrix('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)
    
    # input layer
    l = lasagne.layers.InputLayer(shape=(None, 784),input_var=X)
    
    # hidden layer
    l = lasagne.layers.DenseLayer(l, nonlinearity=lasagne.nonlinearities.identity, num_units=num_units)
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=activation)
    
    # hidden layer
    l = lasagne.layers.DenseLayer(l, nonlinearity=lasagne.nonlinearities.identity, num_units=num_units)
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=activation)

    # output layer
    l = lasagne.layers.DenseLayer(l, nonlinearity=lasagne.nonlinearities.identity,num_units=10)
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=lasagne.nonlinearities.sigmoid)
    
    def loss(t,y):
      return T.mean(T.nnet.binary_crossentropy(y, t))
      # return -T.mean(t*T.log(y)+(1-t)*T.log(1-y))
    
    # THIS IS THE INTERESTING PART
    # adversarial objective
    # as in http://arxiv.org/pdf/1510.04189.pdf
    train_output = lasagne.layers.get_output(l, inputs = X, deterministic=True)
    train_loss = loss(target,train_output)
    adversarial_X = theano.gradient.disconnected_grad(X + 0.08 * T.sgn(theano.gradient.grad(cost=train_loss,wrt=X)))
    train_output = lasagne.layers.get_output(l, inputs = adversarial_X, deterministic=False)
    train_loss = loss(target,train_output)
    
    # Parameters updates
    params = lasagne.layers.get_all_params(l,trainable=True)
    updates = lasagne.updates.adam(loss_or_grads=train_loss, params=params, learning_rate=LR)
    
    # error rate
    test_output = lasagne.layers.get_output(l, deterministic=True)
    test_loss = loss(target,test_output)
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    print('Compiling the graph...')
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([X, target, LR], train_loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([X, target], [test_loss, test_err])

    print('Training...')
    
    custom.train(
            train_fn,val_fn,
            l,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set_X,train_set_t,
            valid_set_X,valid_set_t,
            test_set_X,test_set_t,
            save_path,
            shuffle_parts)
            
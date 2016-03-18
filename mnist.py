
import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle
import gzip

import custom
from collections import OrderedDict

if __name__ == "__main__":
    
    # BN parameters
    batch_size = 32
    print("batch_size = "+str(batch_size))
    
    # MLP parameters
    num_units = 400
    print("num_units = "+str(num_units))
    # max_pooling = 4
    # print("max_pooling = "+str(max_pooling))
    
    # Training parameters
    num_epochs = 150
    print("num_epochs = "+str(num_epochs))
    
    # activation = lasagne.nonlinearities.sigmoid
    # print("activation = lasagne.nonlinearities.sigmoid")
    activation = T.nnet.relu
    print("activation = T.nnet.relu")
    # activation = lasagne.nonlinearities.very_leaky_rectify
    # print("activation = lasagne.nonlinearities.very_leaky_rectify")
    # print("activation = custom.ThresholdedRectifyLayer")
    
    # Decaying LR 
    LR_start = 0.0001
    print("LR_start = "+str(LR_start))
    LR_fin = LR_start
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
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
    
    # for hinge loss
    # train_set_t = 2* train_set_t - 1.
    # valid_set_t = 2* valid_set_t - 1.
    # test_set_t = 2* test_set_t - 1.

    print('Specifying the computations graph...') 
    
    # Prepare Theano variables for inputs and targets
    X = T.matrix('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)
    
    # input layer
    l = lasagne.layers.InputLayer(shape=(None, 784),input_var=X)
    
    # hidden layer
    l = lasagne.layers.DenseLayer(l, nonlinearity=lasagne.nonlinearities.identity, num_units=num_units)
    # l = lasagne.layers.BatchNormLayer(l)
    # l = lasagne.layers.BatchNormLayer(l, alpha=.033) 
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=activation)
    # l = lasagne.layers.DenseLayer(l, nonlinearity=lasagne.nonlinearities.identity, num_units=num_units, b = None)
    # l = custom.ThresholdedRectifyLayer(l)
    # l = lasagne.layers.FeaturePoolLayer(l,pool_size=max_pooling, pool_function=T.max)
    # l0 = l # residual learning stuff
    
    # hidden layer
    # l = lasagne.layers.DenseLayer(l, nonlinearity=lasagne.nonlinearities.identity, num_units=num_units)
    # l = lasagne.layers.NonlinearityLayer(l,nonlinearity=activation)
    # l = lasagne.layers.DenseLayer(l, nonlinearity=lasagne.nonlinearities.identity, num_units=num_units)
    # l = lasagne.layers.NonlinearityLayer(l,nonlinearity=activation)
    
    # hidden layer
    l = lasagne.layers.DenseLayer(l, nonlinearity=lasagne.nonlinearities.identity, num_units=num_units)
    # l = lasagne.layers.BatchNormLayer(l) 
    # l = lasagne.layers.ElemwiseSumLayer([l,l0]) # residual learning stuff
    # l = lasagne.layers.BatchNormLayer(l, alpha=.033) 
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=activation)
    # l = lasagne.layers.DenseLayer(l, nonlinearity=lasagne.nonlinearities.identity, num_units=num_units, b = None)
    # l = custom.ThresholdedRectifyLayer(l)
    # l = lasagne.layers.FeaturePoolLayer(l,pool_size=max_pooling, pool_function=T.max)

    # output layer
    l = lasagne.layers.DenseLayer(l, nonlinearity=lasagne.nonlinearities.identity,num_units=10)
    # l = lasagne.layers.BatchNormLayer(l, alpha=.033) 
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=lasagne.nonlinearities.sigmoid)
    
    def loss(t,y):
      # return T.mean(T.sqr(T.maximum(0.,1.-t*y)))
      return T.mean(T.nnet.binary_crossentropy(y, t))
      # return -T.mean(t*T.log(y)+(1-t)*T.log(1-y))
    
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
    
    print('Compiling theano functions...')
    
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
            
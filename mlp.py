#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""                                   ————————————————————————————————————————————
               ,%%%%%%%%,             Created Time:  -- --th 2018
             ,%%/\%%%%/\%%            Revised Time:  -- --th 2018 
            ,%%%\c "" J/%%%           Contact     :  ------------
   %.       %%%%/ o  o \%%%           Environment :  python2.7
   `%%.     %%%%    _  |%%%           Description :  /
    `%%     `%%%%(__Y__)%%'           Author      :  ___           __ ____   ____
    //       ;%%%%`\-/%%%'                          | | \   ____  / /| |\ \ | |\ \
   ((       /  `%%%%%%%'                            | |\ \ / / | / / | |\ \ | |\ \
    \\    .'          |                             | | \ V /| |/ /  | |\ \| | \ \
     \\  /       \  | |                           _ | |  \_/ |_|_/   |_|\_\|_| \_\
      \\/         ) | |                          \\ | |
       \         /_ | |__                         \ | |
       (___________))))))) 攻城湿                  \_| 
                                     ——————————————————————————————————————————————
"""
from __future__ import division

import numpy as np
from load_data import get_mnist_iter
from activations import sigmoid, sigmoid_backward, relu, relu_backward
from optimizers import sgd
from losses import cross_entropy
import random

def generator(data, labels, batch_size=128, shuffle=True):
    num = len(data)
    indices = range(num)
    if shuffle: 
        random.shuffle(indices)
    for i in range(0, num, batch_size):
        selected_indices = indices[i:min(i+batch_size, num)]
        yield data[selected_indices], labels[selected_indices]
        
def random_next_batch(data, labels, batch_size=128):
    n = len(data)
    idx = np.random.choice(n, batch_size)
    return data[idx], labels[idx]

def fc_forward(x, w, b=None):
    if b is None:
        b = np.zeros([1, w.shape[1]])
    return np.dot(x, w) + b
    
def fc_backward(dz, x, w):
    dx = np.dot(dz, w.T)
    dw = np.dot(x.T, dz)
    db = np.sum(dz, axis=0)
    return dx, dw, db
    
def batch_norm(x, gamma, beta, epsilon=1e-6):
    axis = 0 if len(x.shape)==2 else (0,2,3)
    mean = x.mean(axis=axis, keepdims=True)
    var = ((x-mean)**2).mean(axis=axis, keepdims=True)
    x_hat = (x-mean) / np.sqrt(var+epsilon)
    y = gamma*x_hat + beta
    return y, x_hat, var, mean
    
def batch_norm_backward(dz, x, x_hat, var, mean, gamma, epsilon=1e-6):
    axis = 0 if len(x.shape)==2 else (0,2,3)
    n = len(x)
    center = x - mean
    stable_var = var + epsilon
    dgamma = np.sum(dz*x_hat, axis=axis, keepdims=True)
    dbeta = np.sum(dz, axis=axis, keepdims=True)
    dx_hat = dz * gamma
    dvar = np.sum(dx_hat*center*stable_var**(-3/2)/-2, axis=axis, keepdims=True)
    dmean = np.sum(-dx_hat/np.sqrt(stable_var), axis=axis, keepdims=True) + \
            dvar*(-2)*np.sum(center, axis=axis, keepdims=True)/n
    dx = dx_hat/np.sqrt(stable_var) + dvar*center*2/n + dmean/n
    return dx, dgamma, dbeta
    
def dropout(x, drop_prob=0.5):
    assert 0. <= drop_prob <= 1.0
    if drop_prob == 1.0:
        return np.zeros_like(x)
    keep_prob = 1-drop_prob
    mask = np.random.normal(size=x.shape) < keep_prob
    return x*mask/keep_prob, mask
    
def dropout_backward(dz, mask, drop_prob=0.5):
    return dz*mask/(1-drop_prob)
    
def softmax(x):
    exp = np.exp(x)
    sums = np.sum(exp, axis=-1, keepdims=True)
    return exp/sums
    
def init_params(params):
    params['w1'] = np.random.normal(scale=0.01, size=(784, 256))
#    params['b1'] = np.zeros([256,])
    params['gamma1'] = np.ones([1, 256])
    params['beta1'] = np.zeros([1, 256])
    params['w2'] = np.random.normal(scale=0.01, size=(256, 64))
    params['b2'] = np.zeros([64,])
    params['w3'] = np.random.normal(scale=0.01, size=(64, 10))
    params['b3'] = np.zeros([10,])
        
def forward(x, net, params):
    net['z0'] = x
    net['z1'] = fc_forward(net['z0'], params['w1'])
    net['bn1'], net['x_hat1'], net['var1'], net['mean1'] = batch_norm(net['z1'], params['gamma1'], params['beta1'])
    net['a1'] = sigmoid(net['bn1'])
    net['z2'] = fc_forward(net['a1'], params['w2'], params['b2'])
    net['a2'] = relu(net['z2'])
    net['p2'], net['mask2'] = dropout(net['a2'])
    net['z3'] = fc_forward(net['p2'], params['w3'], params['b3'])
    return softmax(net['z3'])
    
def backward(dz, net, params, grads):
    dz, grads['w3'], grads['b3'] = fc_backward(dz, net['p2'], params['w3'])
    dz = dropout_backward(dz, net['mask2'])
    dz = relu_backward(dz, net['z2'])
    dz, grads['w2'], grads['b2'] = fc_backward(dz, net['a1'], params['w2'])
    dz = sigmoid_backward(dz, net['bn1'])
    dz, grads['gamma1'], grads['beta1'] = batch_norm_backward(dz, net['z1'], net['x_hat1'], 
                                                                net['var1'], net['mean1'], params['gamma1'])
    _, grads['w1'], _ = fc_backward(dz, net['z0'], params['w1'])
    
def cal_acc(y, y_pred):
    true_cls = (y.argmax(axis=-1) == y_pred.argmax(axis=-1)).sum()
    return true_cls/len(y)
    
def train(train_img, train_lbl, val_img, val_lbl, batch_size=16, learning_rate=0.1):
    params = {}
    grads = {}
    net = {}
    init_params(params)
    for step in range(10000):
        x,y = random_next_batch(train_img, train_lbl, batch_size=batch_size)
        x = np.reshape(x, [-1, 784])
        y_pred = forward(x, net, params)
        dz, loss = cross_entropy(y, y_pred)
        backward(dz, net, params, grads)
        sgd(params, grads, learning_rate, batch_size)
        if step%1000 == 0:
            if step > 5000:
                learning_rate *= 0.5
            train_acc = cal_acc(y, y_pred)
            val_x, val_y = random_next_batch(val_img, val_lbl, batch_size=batch_size)
            val_x = np.reshape(val_x, [-1, 784])
            val_acc = cal_acc(val_y, forward(val_x, net, params))
            print 'step:{:d} loss:{:.4f} train_acc:{:.4f} val_acc:{:.4f}'.format(
                                                step, loss.mean(), train_acc, val_acc)
        
if __name__ == '__main__':
    train_img, train_lbl, val_img, val_lbl = get_mnist_iter()
    train(train_img, train_lbl, val_img, val_lbl, 128)
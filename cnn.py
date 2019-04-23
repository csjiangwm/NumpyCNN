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
from activations import relu, relu_backward
from mlp import random_next_batch, cal_acc, fc_forward, fc_backward, softmax, batch_norm, batch_norm_backward
from losses import cross_entropy
from optimizers import sgd

def _padding(data, padding=(1,1)):
    n, c, h, w = data.shape
    outputs = np.zeros(shape=[n, c, h+2*padding[0], w+2*padding[1]])
    outputs[:,:,padding[0]:-padding[0],padding[1]:-padding[1]] = data
    return outputs
    
def _insert_zeros(data, stride=(2,2), r_h=0, r_w=0):
    n,c,h,w = data.shape
    oh = (h-1)*stride[0]+1+r_h
    ow = (w-1)*stride[1]+1+r_w
    outputs = np.zeros([n,c,oh,ow])
    outputs[:,:,::stride[0],::stride[1]] = data
    return outputs
    
def _remove_padding(data, padding=(1,1)):
    return data[:,:,padding[0]:-padding[0],padding[1]:-padding[1]]

def conv(data, weight, bias=None, stride=(1,1), padding=(0,0)):
    if padding[0]>0 or padding[1]>0:
        data = _padding(data, padding)
    N,C,H,W = data.shape
    kn,kc,kh,kw = weight.shape
    if bias is None:
        bias = np.zeros([kn,])
    oh = (H-kh+stride[0])//stride[0]
    ow = (W-kw+stride[1])//stride[1]
    outputs = np.zeros([N, kn, oh, ow])
    for n in range(N):
        for c in range(kn):
            for h in range(oh):
                s_h = stride[0]*h
                e_h = s_h+kh
                for w in range(ow):
                    s_w = stride[1]*w
                    e_w = s_w+kw
                    outputs[n,c,h,w] = np.sum(data[n,:,s_h:e_h,s_w:e_w]*weight[c,:,:,:])+bias[c]
    return outputs
    
def conv_backward(dz, data, weight, stride=(1,1), padding=(0,0)):
    db = np.sum(dz, axis=(0,2,3))
    data = _padding(data, padding) if padding[0]>0 or padding[1]>0 else data
    n,c,h,w = data.shape
    kn,kc,kh,kw = weight.shape
    if stride[0]>1 or stride[1]>1:
        r_h = (h-kh+stride[0])%stride[0]
        r_w = (w-kw+stride[1])%stride[1]
        dz = _insert_zeros(dz, stride, r_h, r_w)
    swap_data = np.swapaxes(data, 0, 1)
    swap_dz = np.swapaxes(dz, 0, 1)
    dw = conv(swap_data, swap_dz).swapaxes(0,1)
    flip_w = weight[:,:,::-1,::-1]
    rot_w = flip_w.swapaxes(0,1)
    padding_dz = _padding(dz, padding=(kh-1,kw-1))
    dx = conv(padding_dz, rot_w)
    dx = _remove_padding(dx, padding) if padding[0]>0 or padding[0]>0 else dx
    return dx, dw, db
    
def max_pooling(data, pooling_size=(2,2), stride=(2,2), padding=(0,0)):
    if padding[0]>0 or padding[1]>0:
        data = _padding(data, padding)
    N,C,H,W = data.shape
    oH = (H-pooling_size[0]+stride[0])//stride[0]
    oW = (W-pooling_size[1]+stride[1])//stride[1]
    outputs = np.zeros([N,C,oH,oW])
    for n in range(N):
        for c in range(C):
            for h in range(oH):
                s_h = h*stride[0]
                e_h = s_h+pooling_size[0]
                for w in range(oW):
                    s_w = w*stride[1]
                    e_w = s_w+pooling_size[1]
                    outputs[n,c,h,w] = np.max(data[n,c,s_h:e_h,s_w:e_w])
    return outputs

def max_pooling_backward(dz, data, pooling_size=(2,2), stride=(2,2), padding=(0,0)):
    if padding[0]>0 or padding[1]>0:
        data = _padding(data, padding)
    outputs = np.zeros_like(data)
    dn,dc,dh,dw = dz.shape
    for n in range(dn):
        for c in range(dc):
            for h in range(dh):
                s_h = h*stride[0]
                e_h = s_h+pooling_size[0]
                for w in range(dw):
                    s_w = w*stride[1]
                    e_w = s_w+pooling_size[1]
                    max_idx = np.argmax(data[n,c,s_h:e_h,s_w:e_w])
                    h_idx = s_h+max_idx//pooling_size[0]
                    w_idx = s_w+max_idx%pooling_size[0]
                    outputs[n,c,h_idx,w_idx] += dz[n,c,h,w]
    if padding[0]>0 or padding[1]>0:
        outputs = _remove_padding(outputs, padding)
    return outputs
    
def flatten(x):
    n = x.shape[0]
    return np.reshape(x, [n,-1])
    
def flatten_backward(dz, x):
    return np.reshape(dz, x.shape)

def init_params(params):
    params['w0'] = np.random.normal(scale=0.01, size=(2,1,5,5))
    params['b0'] = np.zeros([2,])
    params['w1'] = np.random.normal(scale=0.01, size=(3,2,5,5))
#    params['b1'] = np.zeros([3,])
    params['gamma1'] = np.ones([1,3,1,1])
    params['beta1'] = np.zeros([1,3,1,1])
    params['w2'] = np.random.normal(scale=0.01, size=(192, 512))
    params['b2'] = np.zeros([512,])
    params['w3'] = np.random.normal(scale=0.01, size=(512, 10))
    params['b3'] = np.zeros([10,])

def forward(x, net, params):
    net['x']  = x
    net['z0'] = conv(net['x'], params['w0'], params['b0'])
    net['a0'] = relu(net['z0'])
    net['p0'] = max_pooling(net['a0'])
    net['z1'] = conv(net['p0'], params['w1'])
    net['bn1'], net['x_hat1'], net['var1'], net['mean1']=batch_norm(net['z1'], params['gamma1'], params['beta1'])
    net['a1'] = relu(net['bn1'])
    net['fc1']= flatten(net['a1'])
    net['z2'] = fc_forward(net['fc1'], params['w2'], params['b2'])
    net['a2'] = relu(net['z2'])
    net['z3'] = fc_forward(net['a2'], params['w3'], params['b3'])
    return softmax(net['z3'])

def backward(dz, net, params, grads):
    dz, grads['w3'], grads['b3'] = fc_backward(dz, net['a2'], params['w3'])
    dz = relu_backward(dz, net['z2'])
    dz, grads['w2'], grads['b2'] = fc_backward(dz, net['fc1'], params['w2'])
    dz = flatten_backward(dz, net['a1'])
    dz = relu_backward(dz, net['bn1'])
    dz, grads['gamma1'], grads['beta1'] = batch_norm_backward(dz, net['z1'], 
                                                net['x_hat1'], net['var1'], net['mean1'], params['gamma1'])
    dz, grads['w1'], grads['b1'] = conv_backward(dz, net['p0'], params['w1'])
    dz = max_pooling_backward(dz, net['a0'])
    dz = relu_backward(dz, net['z0'])
    _, grads['w0'], grads['b0'] = conv_backward(dz, net['x'], params['w0'])

def train(train_img, train_lbl, val_img, val_lbl, batch_size=64, learning_rate=0.01):
    params = {}
    grads = {}
    net = {}
    init_params(params)
    for step in range(10000):
        x,y = random_next_batch(train_img, train_lbl, batch_size=batch_size)
        y_pred = forward(x, net, params)
        dz, loss = cross_entropy(y, y_pred)
        backward(dz, net, params, grads)
        sgd(params, grads, learning_rate, batch_size)
        if step%1000 == 0:
            if step > 5000:
                learning_rate *= 0.5
            train_acc = cal_acc(y, y_pred)
            val_x, val_y = random_next_batch(val_img, val_lbl, batch_size=batch_size)
            val_acc = cal_acc(val_y, forward(val_x, net, params))
            print 'step:{:d} loss:{:.4f} train_acc:{:.4f} val_acc:{:.4f}'.format(
                                                step, loss.mean(), train_acc, val_acc)

if __name__ == '__main__':
    train_img, train_lbl, val_img, val_lbl = get_mnist_iter()
    train(train_img, train_lbl, val_img, val_lbl, 128)
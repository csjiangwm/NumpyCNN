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

from mlp import *
from losses import *
from activations import *
from cnn import *
import mxnet as mx

def tanh_test():
    data = np.random.normal(size=[1,2,3,3])
    y_pred = tanh(data)
    y = np.ones_like(y_pred)
    grad, loss = mean_square_loss(y, y_pred)
    grad_tanh = tanh_backward(grad, data)
    print grad_tanh
    data = mx.nd.array(data)
    data.attach_grad()
    with mx.autograd.record():
        y_pred = mx.nd.tanh(data)
        l = mx.nd.sum((mx.nd.array(y)-y_pred)**2, axis=0)/2
    l.backward()
    print data.grad
    
def max_pooling_test():
    data = np.random.normal(size=[1,2,4,4])
    y_pred = max_pooling(data)
    y = np.ones_like(y_pred)
    grad, loss = mean_square_loss(y, y_pred)
    grad_pool = max_pooling_backward(grad, data)
    print grad_pool
    data = mx.nd.array(data)
    data.attach_grad()
    with mx.autograd.record():
        y_pred = mx.nd.Pooling(data, kernel=(2,2), stride=(2,2))
        l = mx.nd.sum((mx.nd.array(y)-y_pred)**2, axis=0)/2
    l.backward()
    print data.grad
    
if __name__ == '__main__':
    tanh_test()
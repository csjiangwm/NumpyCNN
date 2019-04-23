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

def relu(x):
    return np.maximum(x, 0)
    
def relu_backward(dz, x):
    return dz * (x>0)
    
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def sigmoid_backward(dz, x):
    return dz * sigmoid(x) * (1-sigmoid(x))
    
def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))
    
def tanh_backward(dz, x):
    return dz * (1-np.square(tanh(x)))
    
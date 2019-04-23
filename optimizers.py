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

def sgd(params, grads, learning_rate, batch_size):
    for key in params.keys():
        params[key][:] -= learning_rate*grads[key]/batch_size
        
def momentum(params, vs, grads, lr, mom, batch_size):
    '''
    initialize vs like params
    '''
    for key in params.keys():
        vs[key][:] = mom*vs[key] + lr*grads[key]/batch_size
        params[key][:] -= vs[key]
        
def RMSprop(params, s, grads, lr, gamma, batch_size, epsilon=1e-6):
    '''
    initialize s like params
    '''
    for key in params.keys():
        g = grads[key]/batch_size
        s[key][:] = gamma*s[key] + (1-gamma)*g*g
        g_ = lr/np.sqrt(s[key]+epsilon) * g
        params[key][:] -= g_
        
def Adam(params, vs, s, grads, batch_size, t, lr, beta1=0.9, beta2=0.999, epsilon=1e-6):
    for key in params.keys():
        g = grads[key]/batch_size
        vs[key][:] = beta1*vs[key] + (1-beta1)*g
        s[key][:] = beta2*s[key] + (1-beta2)*g*g
        vs_ = vs[key]/(1-beta1**t)
        s_ = s[key]/(1-beta2**t)
        g_ = lr*vs_/np.sqrt(s_+epsilon)
        params[key][:] -= g_ 
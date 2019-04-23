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
import numpy as np
import gzip, struct

def read_data(label, image):
    """
    download and read data into numpy
    """
    with gzip.open(label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.frombuffer(flbl.read(), dtype=np.int8)
    with gzip.open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def get_mnist_iter():
    (train_lbl, train_img) = read_data(
            'MNIST_data/train-labels-idx1-ubyte.gz', 'MNIST_data/train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
            'MNIST_data/t10k-labels-idx1-ubyte.gz', 'MNIST_data/t10k-images-idx3-ubyte.gz')
    return to4d(train_img), one_hot(train_lbl,10), to4d(val_img), one_hot(val_lbl,10)
    
def one_hot(lbl, ncls):
    return (np.arange(ncls)==lbl[:,None]).astype(np.integer)
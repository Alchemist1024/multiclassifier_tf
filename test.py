# -*- encoding: utf-8 -*-
'''
@File        :test.py
@Time        :2021/01/19 10:29:42
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import tensorflow as tf

a = tf.constant([0.1, 0.2, 0.6, 0.9], dtype=tf.float32)
b = a > 0.5
with tf.Session() as sess:
    # sess.run(a)
    # sess.run(b)
    print(a)
    print(b)
# -*- encoding: utf-8 -*-
'''
@File        :BertForMultiClassification.py
@Time        :2021/01/19 16:15:47
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import bert
from bert import run_classifier, optimization, modeling
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.9
session = tf.Session(config=config)


class Model:
    def __init__(self, config):
        self.bert_config = config.bert_config
        self.num_labels = config.num_labels
        self.use_one_hot_embeddings = config.use_one_hot_embeddings

    def callback(self, input_ids, input_mask, segment_ids, labels, is_training):
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings)

        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            'output_weights', [self.num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        output_bias = tf.get_variable(
            'output_bias', [self.num_labels], initializer=tf.zeros_initializer()
        )

        with tf.variable_scope('loss'):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            probabilities = tf.nn.sigmoid(logits)
            prediction = tf.cast(probabilities > 0.5, dtype=tf.int32)

            labels = tf.cast(labels, tf.float32)
            labels = tf.reshape(labels, [-1, self.num_labels])

            tf.logging.info(f"num_labels:{self.num_labels};logits:{logits};label:{labels}")
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(per_example_loss)

            return loss, per_example_loss, prediction, probabilities

# -*- encoding: utf-8 -*-
'''
@File        :train_.py
@Time        :2021/01/20 19:37:50
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import os
from bert import modeling, optimization
import tensorflow as tf
from config import Config
from BertForMultiClassification import Model
from DataLoader import MultiClassificationProcessor, Features2TF
import random

class Trainer:
    def __init__(self, config):
        self.model = Model(config)
        self.processor = MultiClassificationProcessor(config.class_path)
        self.feature2tf = Features2TF()

        self.bert_config = config.bert_config
        self.init_checkpoint = config.bert_init_checkpoint
        self.tokenizer = config.tokenizer

        self.num_labels = config.num_labels
        self.learning_rate = config.learning_rate
        self.num_labels = config.num_labels
        self.use_tpu = config.use_tpu
        self.use_one_hot_embeddings = config.use_one_hot_embeddings
        self.batch_size = config.batch_size
        self.num_train_epochs = config.num_epochs
        self.save_checkpoints_steps = config.save_checkpoints_steps
        self.save_summary_steps = config.save_summary_steps
        self.warmup_proportion = config.warmup_proportion
        self.max_seq_length = config.max_seq_length

        self.output_dir = config.output_dir
        self.pb_dir = config.pb_dir
        self.train_path = config.train_path
        self.dev_path = config.dev_path

    def model_fn_builder(self, num_train_steps, num_warmup_steps):
        def model_fn(features, labels, mode, params):
            input_ids = features['input_ids']
            input_mask = features['input_mask']
            segment_ids = features['segment_ids']
            label_ids = features['label_ids']

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            # (total_loss, per_example_loss, logits, probabilities) = self.model.callback(
            #     input_ids, input_mask, segment_ids, label_ids, is_training)
            (total_loss, per_example_loss, predictions, probabilities) = self.model.callback(
                input_ids, input_mask, segment_ids, label_ids, is_training)

            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            scaffold_fn = None

            if self.init_checkpoint:
                (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                    tvars, self.init_checkpoint)
                if self.use_tpu:
                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
                        return tf.train.Scaffold()
                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"

            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(
                    total_loss, self.learning_rate, num_train_steps, num_warmup_steps, self.use_tpu)

                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold=scaffold_fn)

            elif mode == tf.estimator.ModeKeys.EVAL:
                def metric_fn(per_example_loss, label_ids, probabilities):
                    # precision = tf.metrics.precision(labels=label_ids, predictions=label_ids)
                    # recall = tf.metrics.recall(labels=label_ids, predictions=probabilities)
                    # f1 = (2 * precision[0] * recall[0] / (precision[0] + recall[0]),recall[1])
                    # accuracy = tf.metrics.accuracy(
                    #     labels=label_ids, predictions=probabilities)
                    # loss = tf.metrics.mean(values=per_example_loss)
                    # return {
                    #     "eval_accuracy": accuracy,
                    #     "eval_precision": precision,
                    #     "eval_recall": recall,
                    #     "eval_f1": f1,
                    #     "eval_loss": loss,
                    # }
                    logits_split = tf.split(probabilities, self.num_labels, axis=-1)
                    label_ids_split = tf.split(label_ids, self.num_labels, axis=-1)
                    eval_dict = {}
                    for j, logits in enumerate(logits_split):
                        label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                        current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                        eval_dict[str(j)] = (current_auc, update_op_auc)
                    eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                    return eval_dict
                # def metric_fn(per_example_loss, label_ids, predictions):
                #     precision = tf.metrics.precision(labels=label_ids, predictions=predictions)
                #     loss = tf.metrics.mean(values=per_example_loss)
                #     return {
                #         'eval_precision': precision,
                #         'eval_loss': loss,
                #     }

                eval_metrics = metric_fn(per_example_loss, label_ids, probabilities)
                # eval_metrics = metric_fn(per_example_loss, label_ids, predictions)

                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=eval_metrics,
                    scaffold=scaffold_fn)

            else:
                print('mode:', mode, 'probabilities:', probabilities)
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={'probabilities': probabilities},
                    scaffold=scaffold_fn)
            return output_spec
        return model_fn


    def run(self, mode='train'):
        def serving_input_fn():
            label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
            input_ids = tf.placeholder(tf.int32, [None, self.max_seq_length], name='input_ids')
            input_mask = tf.placeholder(tf.int32, [None, self.max_seq_length], name='input_mask')
            segment_ids = tf.placeholder(tf.int32, [None, self.max_seq_length], name='segment_ids')
            input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
                'label_ids': label_ids,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            })()
            return input_fn

        run_config = tf.estimator.RunConfig(
            model_dir=self.output_dir,
            save_summary_steps=self.save_summary_steps,
            keep_checkpoint_max=1,
            save_checkpoints_steps=self.save_checkpoints_steps)

        train_examples = self.processor.get_train_examples(self.train_path)
        num_train_steps = int(len(train_examples) /
                              self.batch_size * self.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)
        rng = random.Random(1)
        rng.shuffle(train_examples)

        model_fn = self.model_fn_builder(num_train_steps, num_warmup_steps)
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={'batch_size': self.batch_size})

        early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='eval_loss',
            max_steps_without_decrease=6*self.save_checkpoints_steps,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=self.save_checkpoints_steps)

        train_file = os.path.join(self.output_dir, 'train.tf_record')
        if not os.path.exists(train_file):
            self.feature2tf.file_based_convert_examples_to_features(
                train_examples, self.max_seq_length, self.tokenizer, train_file)
        train_input_fn = self.feature2tf.file_based_input_fn_builder(
            input_file=train_file,
            seq_length=self.max_seq_length,
            num_labels=self.num_labels,
            is_training=True,
            drop_remainder=True
        )

        eval_examples = self.processor.get_dev_examples(self.dev_path)
        eval_file = os.path.join(self.output_dir, 'eval.tf_record')
        if not os.path.exists(eval_file):
            self.feature2tf.file_based_convert_examples_to_features(
                eval_examples, self.max_seq_length, self.tokenizer, eval_file)
        eval_input_fn = self.feature2tf.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.max_seq_length,
            num_labels=self.num_labels,
            is_training=False,
            drop_remainder=False
        )

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=num_train_steps, hooks=[early_stopping_hook])
        exporter = tf.estimator.BestExporter(
            serving_input_receiver_fn=serving_input_fn,
            exports_to_keep=2)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn, steps=None, exporters=exporter)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        estimator._export_to_tpu = False
        estimator.export_saved_model(self.pb_dir, serving_input_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    config = Config('data')
    app = Trainer(config)
    app.run()

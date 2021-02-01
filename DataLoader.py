# -*- encoding: utf-8 -*-
'''
@File        :DataLoader.py
@Time        :2021/01/19 14:15:45
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import os
import tensorflow as tf
from collections import OrderedDict


class InputExample:
    def __init__(self, text_a, text_b=None, labels=None):
        # self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.is_real_example = is_real_example


class DataProcessor:
    def get_train_examples(self, data_path):
        raise NotImplementedError()

    def get_dev_examples(self, data_path):
        raise NotImplementedError

    def get_test_examples(self, data_path):
        raise NotImplementedError
    
    def get_labels(self, data_path):
        raise NotImplementedError


class MultiClassificationProcessor(DataProcessor):
    '''自定义的多标签和多分类的数据处理类，主要是获得句子和相应标签。
    '''
    def __init__(self, label_path):
        self.labels = [label.strip() for label in open(label_path, 'r', encoding='utf-8')]

    def _create_examples(self, data, labels_available=True):
        examples = []
        for record in data:
            labels = []
            try:
                label, text_a = record.split('\t')
            except Exception:
                continue
            label_idx = [0] * len(self.labels)
            if labels_available:
                for sub_lab in label.split(','):
                    idx = self.labels.index(sub_lab)
                    label_idx[idx] = 1
            examples.append(InputExample(text_a=text_a, labels=label_idx))
        return examples

    def _create_single_example(self, sen):
        '''将一个句子转化为example
        '''
        label_ids = [0] * len(self.labels)
        text_a = sen
        example = InputExample(text_a=text_a, labels=label_ids)
        return example

    def get_train_examples(self, data_path, size=-1):
        data = []
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                data.append(record.strip())
        if size == -1:
            return self._create_examples(data)
        else:
            import random
            random.seed(1)
            data_sample = random.sample(data, size)
            return self._create_examples(data_sample)

    def get_dev_examples(self, data_path, size=-1):
        data = []
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                data.append(record.strip())
        if size == -1:
            return self._create_examples(data)
        else:
            import random
            random.seed(1)
            data_sample = random.sample(data, size)
            return self._create_examples(data_sample)

    def get_test_examples(self, data_path, size=-1):
        data = []
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                data.append(record.strip())
        if size == -1:
            return self._create_examples(data)
        else:
            import random
            random.seed(1)
            data_sample = random.sample(data, size)
            return self._create_examples(data_sample)

    def get_labels(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as reader:
            for record in reader:
                data.append(record.strip())
        return data


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    '''
    对于句子对形式进行截断的时候，每次对长的句子进行截断，因为短的句子携带更多的信息。
    '''
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for ex_idx, example in enumerate(examples):
        print(ex_idx)
        # if ex_idx > 100:
        #     break
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)
        else:
            if len(tokens_a) > max_seq_length-2:
                tokens_a = tokens_a[:max_seq_length-2]

        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            segment_ids += [1] * (len(tokens_b)+1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length-len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        labels_ids = []
        for label in example.labels:
            labels_ids.append(int(label))

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask = input_mask,
                                      segment_ids=segment_ids,
                                      label_ids=labels_ids))
    return features


def convert_single_example(example, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)
    else:
        if len(tokens_a) > max_seq_length-2:
            tokens_a = tokens_a[:max_seq_length-2]

    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    segment_ids = [0] * len(tokens)
    
    if tokens_b:
        tokens += tokens_b + ['[SEP]']
        segment_ids += [1] * (len(tokens_b)+1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length-len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    labels_ids = []
    for label in example.labels:
        labels_ids.append(int(label))

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_ids=labels_ids)
    return feature


class Features2TF:
    '''特征存入TFRecord file中，并从其中读取，加速数据加载过程。
    '''
    def __init__(self):
        pass
    
    def file_based_convert_examples_to_features(self, examples, max_seq_length, tokenizer, output_file):
        writer = tf.python_io.TFRecordWriter(output_file)

        for example in examples:
            feature = convert_single_example(example, max_seq_length, tokenizer)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = OrderedDict()
            features['input_ids'] = create_int_feature(feature.input_ids)
            features['input_mask'] = create_int_feature(feature.input_mask)
            features['segment_ids'] = create_int_feature(feature.segment_ids)
            if isinstance(feature.label_ids, list):
                label_ids = feature.label_ids
            else:
                label_ids = feature.label_ids[0]
            features['label_ids'] = create_int_feature(label_ids)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()

    def file_based_input_fn_builder(self, input_file, seq_length, num_labels, is_training, drop_remainder):
        name_to_features = {
            'input_ids': tf.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.FixedLenFeature([num_labels], tf.int64)
        }

        def _decode_record(record, name_to_features):
            example = tf.parse_single_example(record, name_to_features)

            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
            return example

        def input_fn(params):
            batch_size = params['batch_size']
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder
                )
            )
            return d

        return input_fn

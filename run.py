# -*- encoding: utf-8 -*-
'''
@File        :run.py
@Time        :2021/01/21 08:47:01
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
import numpy as np

from config import Config
from DataLoader import convert_single_example, MultiClassificationProcessor


class Predictor:
    def __init__(self, config, pb_dir, label_path):
        self.config = config
        self.pb_dir = pb_dir
        self.prediction_fn = tf.contrib.predictor.from_saved_model(self.pb_dir)
        self.processor = MultiClassificationProcessor(label_path)
        self.labels = [label.strip() for label in open(label_path, 'r', encoding='utf-8')]
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}

    def run(self, sen, thresh=0.5):
        '''支持batch，先写单个
        '''
        example = self.processor._create_single_example(sen)
        feature = convert_single_example(example, self.config.max_seq_length, self.config.tokenizer)

        prediction = self.prediction_fn({
            'input_ids': np.array(feature.input_ids).reshape(-1, self.config.max_seq_length),
            'input_mask': np.array(feature.input_mask).reshape(-1, self.config.max_seq_length),
            'segment_ids': np.array(feature.segment_ids).reshape(-1, self.config.max_seq_length),
        })
        eval_logits = prediction['probabilities'][0]
        eval_logits = eval_logits.tolist()

        scores = {}
        for idx, score in enumerate(eval_logits):
            scores[idx] = score
        scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        f_idx = []
        f_scores = []

        for i in range(len(scores_sorted)):
            if scores_sorted[i][1] > thresh:
                f_idx.append(scores_sorted[i][0])
                f_scores.append(scores_sorted[i][1])
        labels = []
        for idx in f_idx:
            labels.append(self.id2label[idx])

        return labels, f_scores

    def run_with_top2(self, sen, thresh=0.5):
        example = self.processor._create_single_example(sen)
        feature = convert_single_example(example, self.config.max_seq_length, self.config.tokenizer)

        prediction = self.prediction_fn({
            'input_ids': np.array(feature.input_ids).reshape(-1, self.config.max_seq_length),
            'input_mask': np.array(feature.input_mask).reshape(-1, self.config.max_seq_length),
            'segment_ids': np.array(feature.segment_ids).reshape(-1, self.config.max_seq_length),
        })
        eval_logits = prediction['probabilities'][0]
        eval_logits = eval_logits.tolist()

        scores = {}
        for idx, score in enumerate(eval_logits):
            scores[idx] = score
        scores_sorted = sorted(scores.ites(), key=lambda x: x[1], reverse=True)
        f_idx = []
        f_scores = []

        for i in range(2):
            if scores_sorted[i][1] > thresh:
                f_idx.append(scores_sorted[i][0])
                f_scores.append(scores_sorted[i][1])
        labels = []
        for idx in f_idx:
            labels.append(self.id2label[idx])
 
        return labels, f_scores


def evaluate(data_path):
    '''在测试集上对模型进行评估，自定义评估方式
    '''
    config = Config('data')
    pb_dir = 'data/pb_dir/1611222901'
    label_path = 'data/labels_part1.txt'
    predict = Predictor(config, pb_dir, label_path)

    true_cnt = 0
    total_cnt = 0
    recall_cnt = 0

    all_data = []
    wrong_data = []
    with open(data_path, 'r', encoding='utf-8') as reader:
        for record in reader:
            try:
                label, data = record.strip().split('\t')
            except Exception:
                continue

            pre_labels, _ = predict.run_with_top2(data, 0.5)
            true_labels = label.split(',')

            if set(pre_labels).issubset(set(true_labels)) and len(pre_labels) > 0:
                true_cnt += 1
                total_cnt += 1
                recall_cnt += 1
            elif len(pre_labels) > 0:
                total_cnt += 1
                recall_cnt += 1
                wrong_data.append(data + '####' + ','.join(true_labels) + '####' + ','.join(pre_labels))
            else:
                total_cnt += 1

            print(f"第{total_cnt}条数据，匹配准确率:{true_cnt/(recall_cnt+1) * 100: .2f}%，匹配率:{recall_cnt/total_cnt * 100:.2f}%，整体准确率:{true_cnt/total_cnt * 100:.2f}%")

    return wrong_data


def model_test(data_path):
    '''对模型进行评估
    '''
    config = Config('data')
    pb_dir = 'data/pb_dir/1611286694'
    # pb_dir = 'data/checkpoint_dir/export/best_exporter/1611284778'
    label_path = 'data/labels_part1.txt'
    predict = Predictor(config, pb_dir, label_path)

    true_cnt = 0
    total_cnt = 0
    recall_cnt = 0

    wrong_data = []
    with open(data_path, 'r', encoding='utf-8') as reader:
        for record in reader:
            try:
                label, data = record.strip().split('\t')
            except Exception:
                continue

            pre_labels, _ = predict.run(data, 0.5)
            true_labels = label.split(',')

            if set(pre_labels) == set(true_labels):
                true_cnt += 1
                total_cnt += 1
            else:
                total_cnt += 1
                wrong_data.append(data + '####' + ','.join(true_labels) + '####' + ','.join(pre_labels))

            print(f"第{total_cnt}条数据，准确率:{true_cnt/total_cnt * 100: .2f}%") # 

    return wrong_data


if __name__ == '__main__':
    config_ = Config('data')
    pb_dir = 'data/pb_dir/1611286694'
    label_path = 'data/labels_part1.txt'

    data_path = 'data/test_part1.txt'
    model_test(data_path)
    # predictor = Predictor(config_, pb_dir, label_path)

    # sen = '题干:一个西瓜约重5(  ).A.克B.千克C.米 解:一个西瓜重约5千克；故选：B．分析:根据生活经验、对质量单位和数据大小的认识，可知计量一个西瓜重量，应用质量单位，结合数据可知：应用“千克”做单位，据此解答．注释:无'

    # labels, scores = predictor.run(sen)

    # print(labels)
    # print(scores)
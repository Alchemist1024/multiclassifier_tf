# -*- encoding: utf-8 -*-
'''
@File        :config.py
@Time        :2020/12/15 08:39:03
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

from bert import tokenization, modeling


class Config:
    '''配置参数
    '''
    def __init__(self, dataset):
        self.output_dir = 'data/checkpoint_dir'
        self.pb_dir = 'data/pb_dir'
        self.train_path = dataset + '/train_part1.txt'
        self.dev_path = dataset + '/test_part1.txt'
        self.test_path = dataset + '/test_part1.txt'
        self.class_path = dataset + '/labels_part1.txt'

        self.bert_config_path = './chinese_wwm_L-12_H-768_A-12/bert_config.json'
        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_path)
        self.bert_init_checkpoint = './chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
        self.bert_vocab = './chinese_wwm_L-12_H-768_A-12/vocab.txt'
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.bert_vocab)

        self.require_improment = 1000
        self.use_one_hot_embeddings = False
        self.warmup_proportion = 0.1
        self.use_tpu = False
        self.num_epochs = 20
        self.batch_size = 8
        self.max_seq_length = 256
        self.learning_rate = 5e-5
        self.save_checkpoints_steps = 1000
        self.save_summary_steps = 500
        self.labels = [label.strip() for label in open(self.class_path)]
        self.num_labels = len(self.labels)
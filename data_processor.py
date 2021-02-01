# -*- encoding: utf-8 -*-
'''
@File        :data_processor.py
@Time        :2021/01/07 17:27:40
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import re
import pandas as pd
import math
import numpy as np
import random
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from typing import List, Text


class Processor:
    def __init__(self):
        pass

    def filter(self, text: str) -> str:
        '''数据过滤
        '''
        del_list = ['\n', '\t', '\r', u'\u3000', u'\ufeff', u'\xa0']
        text = BeautifulSoup(text, 'lxml').text
        sen = ''
        for char in text:
            if char not in del_list:
                sen += char
        sen = re.sub('&lt;', '<', sen)
        sen = re.sub('&gt;', '>', sen)
        sen = re.sub('nan', '无', sen)
        sen = re.sub('(  )+', ' ', sen)
        return sen

    def preprocess(self, data_path):
        data_frame = pd.read_excel(data_path)
        data = data_frame.values
        all_data = []
        cnt = 0
        for record in data:
            cnt += 1
            print(f"第{cnt}条记录...")
            label, content, solution, analysis, comment, answer = str(record[2]), str(
                record[3]), str(record[4]), str(record[5]), str(record[6]), str(record[7]) #存在nan现象

            content = self.filter(content).strip()
            solution = self.filter(solution).strip()
            analysis = self.filter(analysis).strip()
            comment = self.filter(comment).strip()
            answer = self.filter(answer).strip() # 这一部分可以不要，尝试一下
            if solution.startswith('解：'):
                solution = re.sub('解：', '解:', solution)
            else:
                solution = '解:' + solution
            # new_record = label + '\t' + '题干:' + content + ' 分析:' + \
            #     analysis + ' ' + explain + ' 注释:' + comment + ' 答案:' + answer
            new_record = label + '\t' + '题干:' +  content + ' ' + solution + '分析:' + analysis + '注释:' + comment 
            all_data.append(new_record)
        return all_data

    def split_dataset(self, data: List[Text]):
        '''没有办法分开，可以把两个标签当成单个标签
        '''
        dataSet = []
        labels = []
        for record in data:
            label, content = record.split('\t')
            dataSet.append(content)
            labels.append(label)
        X_train, X_test, y_train, y_test = train_test_split(dataSet, labels, test_size=0.1, random_state=0)
        train = []
        test = []
        for x, y in zip(X_train, y_train):
            train.append(y + '\t' + x)
        for x, y in zip(X_test, y_test):
            test.append(y + '\t' + x)
        return train, test

    def get_sub_dataset(self, data_path, label_info_path, sub_classes=['统计与概率', '数学广角', '奥数-规律']):
        '''按照大类获取数据集
        '''
        label_info = pd.read_excel(label_info_path).values
        labels = []
        for info in label_info:
            top_label, label = info[1], info[3]
            for sub_class in sub_classes:
                if top_label.startswith(sub_class):
                    labels.append(label)
                    break

        all_data = self.preprocess(data_path)
        data_part = []

        for data in all_data:
            label, _ = data.split('\t')
            label_list = label.split(',')
            if set(label_list).issubset(set(labels)):
                data_part.append(data)

        train, test = self.split_dataset(data_part)
        return train, test, labels

    def get_dataset(self, data_path, label_info_path):
        label_info = pd.read_excel(label_info_path).values
        labels = []
        for info in label_info:
            label = info[3]
            labels.append(label)

        all_data = self.preprocess(data_path)

        train, test = self.split_dataset(all_data)
        return train, test, labels

    def persist(self, data_path, train_path, test_path, label_path, label_info_path, flag=True):
        if flag:
            train, test, labels = self.get_dataset(data_path, label_info_path)
        else:
            train, test, labels = self.get_sub_dataset(data_path, label_info_path)

        with open(train_path, 'w', encoding='utf-8') as writer:
            for data in train:
                writer.write(data + '\n')

        with open(test_path, 'w', encoding='utf-8') as writer:
            for data in test:
                writer.write(data + '\n')

        with open(label_path, 'w', encoding='utf-8') as writer:
            for label in labels:
                writer.write(label + '\n')
        print('save successfully!')


if __name__ == '__main__':
    # data_path = 'data/all_data.xlsx'
    # train_path = 'data/train_all.txt'
    # test_path = 'data/test_all.txt'
    # label_path = 'data/labels_all.txt'
    # process = Processor()
    # process.persist(data_path, train_path, test_path, label_path)

    # data_path = 'data/all_data.xlsx'
    # label_info_path = 'data/label_info.xlsx'
    # process = Processor()
    # _, _, labels = process.get_sub_dataset(data_path, label_info_path)
    # to_path = 'label_test.txt'
    # with open(to_path, 'w', encoding='utf-8') as writer:
    #     for label in labels:
    #         writer.write(label + '\n')
    # data_path = 'data/all_data.xlsx'
    # label_info_path = 'data/label_info.xlsx'
    # train_path = 'data/train_part1.txt'
    # test_path = 'data/test_part1.txt'
    # label_path = 'data/labels_part1.txt'

    process = Processor()
    # process.persist(data_path, train_path, test_path, label_path, label_info_path, False)
    sen = '解：成本价：124.8<math title=\div xmlns="http://www.w3.org/1998/Math/MathML"><mo>÷</mo></math><SPAN> 【（1—20%）<math title=\times xmlns="http://www.w3.org/1998/Math/MathML"><mo>×</mo></math><SPAN> （1+30%）】=120（元） 因为124.8&gt;120，所以这种商品卖出一件是赚了，赚的钱是124.8—120=4.8（元）。'
    process.filter(sen)
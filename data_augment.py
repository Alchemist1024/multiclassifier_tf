# -*- encoding: utf-8 -*-
'''
@File        :data_augment.py
@Time        :2020/12/29 09:03:35
@Author      :xiaoqifeng
@Version     :1.0
@Contact:    :unknown
'''

import synonyms
import random
from random import shuffle
import jieba
import logging
random.seed(1)
jieba.setLogLevel(logging.INFO)


class EasyDataAugment:
    def __init__(self, stop_words_path):
        self.stop_words = [word.strip() for word in open(stop_words_path, 'r', encoding='utf-8')]

    def clean(self, text):
        '''数据清理
        '''
        text = text.replace(' ', '')
        chars = ['\t', '\n', '\r', u'\u3000', u'\ufeff', u'\xa0']
        for char in chars:
            text.replace(char, '')
        return text

    def synonym_replacement(self, words, n):
        '''同义词替换
        '''
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonym_words, _ = synonyms.nearby(random_word)
            if len(synonym_words) >= 1:
                synonym = random.choice(synonym_words)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        
        sen = ' '.join(new_words)
        new_words = sen.split(' ')

        return new_words

    def random_deletion(self, words, p):
        '''随机删除
        '''
        if len(words) == 1:
            return words

        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)
        
        if len(new_words) == 0:
             rand_int = random.randint(0, len(words)-1)
             return [words[rand_int]]

        return new_words

    def random_swap(self, words, n):
        '''随机交换
        '''
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words

    def random_insertion(self, words, n):
        '''随机插入
        '''
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        synonym_words = []
        counter = 0
        while len(synonym_words) < 1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonym_words, _ = synonyms.nearby(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonym_words[0]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)

    def segment_shuffle(self, sen):
        '''把句子分段shuffle，暂时先不用
        '''
        new_sen = sen.copy()
        new_sen = new_sen[-100: -1] + new_sen[0: 100] + new_sen[100:]
        return new_sen

    def run(self, sen, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
        sen = self.clean(sen)
        words = jieba.lcut(sen)
        num_words = len(words)

        augmented_sens = []
        num_new_per_technique = int(num_aug/4) + 1

        if alpha_sr > 0:
            n_sr = max(1, int(alpha_sr*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.synonym_replacement(words, n_sr)
                augmented_sens.append(''.join(a_words))

        if alpha_ri > 0:
            n_ri = max(1, int(alpha_ri*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_insertion(words, n_ri)
                augmented_sens.append(''.join(a_words))

        if alpha_rs > 0:
            n_rs = max(1, int(alpha_rs*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_swap(words, n_rs)
                augmented_sens.append(''.join(a_words))

        if p_rd > 0:
            for _ in range(num_new_per_technique):
                a_words = self.random_deletion(words, p_rd)
                augmented_sens.append(''.join(a_words))
        
        augmented_sens = [self.clean(sen) for sen in augmented_sens]
        shuffle(augmented_sens)

        if num_aug >= 1:
            augmented_sens = augmented_sens[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sens)
            augmented_sens = [s for s in augmented_sens if random.uniform(0, 1) < keep_prob]

        augmented_sens.append(sen)

        return augmented_sens


def augment(data_path, to_path):
    data_augmentation = []
    eda = EasyDataAugment('data/stopwords.txt')
    cnt = 0

    print('-' * 20 + '开始数据增强' + '-' * 20)
    with open(data_path, 'r', encoding='utf-8') as reader:
        for record in reader:
            label, data = record.strip().split('\t')
            candidates = eda.run(data, num_aug=9)
            for candidate in candidates:
                cnt += 1
                print(f"第{cnt}条记录...")
                new_data = label + '\t' + candidate
                data_augmentation.append(new_data)
    print('-' * 20 + '数据增强完成' + '-' * 20)

    print('-' * 20 + '数据保存' + '-' * 20)
    with open(to_path, 'w', encoding='utf-8') as writer:
        for data in data_augmentation:
            print(data)
            writer.write(data + '\n')

    print('-' * 20 + '数据保存完成' + '-' * 20)


if __name__ == '__main__':
    # sen = '甲数比乙数多20％,就是乙数比甲数少20％.  掌握求一个数比另一个数多或少百分之几的应用题的方法：先求出多或少多少,再除以单位“1”,要注意找准单位“1”.求甲数比乙数多百分之几,把多的除以乙数；求乙数比甲数少百分之几,把少的除以甲数；这两题要注意单位“1”不同.'
    # eda = EasyDataAugment('data/stopwords.txt')
    # res = eda.run(sen)
    # for i in res:
    #     print(i)
    data_path = 'data/train_all.txt'
    to_path = 'data/train_all_augmentation.txt'
    augment(data_path, to_path)
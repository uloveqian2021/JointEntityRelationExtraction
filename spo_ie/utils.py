#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
# import unicodedata
from tqdm import tqdm
import logging as log
log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO)


# 加载数据集合
def load_data(file_name, is_train=False, debug=False):
    D = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            l = json.loads(line)
            if is_train:
                text = l['text']
            else:
                text = l['text']
            D.append({'text': text,
                      'spo_list': [(spo['subject'], spo['predicate'], spo['object']) for spo in l['spo_list']]
                      })
    if debug:
        return D[:len(D)//4]
    return D


def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class DataGenerator(object):
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d


class data_generator_v0(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False, predicate2id=None, max_len=None, tokenizer=None):
        batch_text, batch_token_ids, batch_mask, batch_segment_ids = [], [], [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens_ids, segment_ids = tokenizer.encode(d['text'], maxlen=max_len)
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, tokens_ids)
                o_idx = search(o, tokens_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                input_mask = [1] * len(tokens_ids)    # TODO
                # segment_ids = [0] * len(token_ids)
                subject_labels = np.zeros((len(tokens_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(tokens_ids), len(predicate2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append(tokens_ids)
                batch_text.append(d['text'])
                batch_mask.append(input_mask)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids, max_len)
                    batch_mask = sequence_padding(batch_mask, max_len)
                    batch_segment_ids = sequence_padding(batch_segment_ids, max_len)
                    batch_subject_labels = sequence_padding(
                        batch_subject_labels, max_len
                    )
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_object_labels, max_len)
                    yield batch_text, batch_token_ids, batch_mask, batch_segment_ids, batch_subject_labels, \
                          batch_subject_ids, batch_object_labels

                    batch_text, batch_token_ids, batch_mask, batch_segment_ids = [], [], [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False, predicate2id=None, max_len=None, tokenizer=None):
        T0, T1, T2, M1, S1, S2, K3, O1, O2 = [], [], [], [], [], [], [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=max_len)
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # mask
                input_mask = [1] * len(token_ids)  # TODO

                s1, s2 = np.zeros(len(token_ids)), np.zeros(len(token_ids))
                for j in spoes:
                    s1[j[0]] = 1
                    s2[j[1]] = 1
                # 随机选一个subject
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                o1 = np.zeros((len(token_ids), len(predicate2id)))
                o2 = np.zeros((len(token_ids), len(predicate2id)))
                for o in spoes.get(subject_ids, []):
                    o1[o[0]][o[2]] = 1
                    o2[o[1]][o[2]] = 1
                # 构建batch
                T0.append(d['text'])
                T1.append(token_ids)
                T2.append(segment_ids)
                M1.append(input_mask)
                S1.append(s1)
                S2.append(s2)
                K3.append(subject_ids)
                # K1.append([start])
                # K2.append([end])
                O1.append(o1)
                O2.append(o2)
                if len(T1) == self.batch_size or is_end:
                    T1 = sequence_padding(T1)
                    T2 = sequence_padding(T2)
                    M1 = sequence_padding(M1)
                    S1 = sequence_padding(S1)
                    S2 = sequence_padding(S2)
                    O1 = sequence_padding(O1)
                    O2 = sequence_padding(O2)
                    # K1, K2, K3 = np.array(K1), np.array(K2), np.array(K3)
                    K3 = np.array(K3)
                    yield T0, T1, M1, T2,  S1, S2, K3, O1, O2
                    T0, T1, T2, M1, S1, S2, K3, O1, O2, = [], [], [], [], [], [], [], [], []


def closed_single(text, model, sess, tokenizer, max_len, id2predicate):
    tokens = tokenizer.tokenize(text, maxlen=max_len)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=max_len)
    input_mask = [1] * len(token_ids)
    token_ids = sequence_padding([token_ids], max_len)
    input_mask = sequence_padding([input_mask], max_len)
    segment_ids = sequence_padding([segment_ids], max_len)
    feed_dict = {model.char_inputs: token_ids,
                 model.mask_inputs: input_mask,
                 model.segment_ids: segment_ids,
                 model.is_training: False}
    subject_s1_preds, subject_s2_preds = sess.run([model.prob_s1, model.prob_s2], feed_dict)
    start = np.where(subject_s1_preds[0] > 0.5)[0]
    end = np.where(subject_s2_preds[0] > 0.45)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        input_mask = np.repeat(input_mask, len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        feed_dict = {model.char_inputs: token_ids,
                     model.mask_inputs: input_mask,
                     model.segment_ids: segment_ids,
                     model.tag_s_ids: subjects,
                     model.is_training: False}
        object_o1_preds, object_o2_preds = sess.run([model.prob_o1, model.prob_o2], feed_dict)
        for subject, object_pred1, object_pred2 in zip(subjects, object_o1_preds, object_o2_preds):
            start = np.where(object_pred1 > 0.5)
            end = np.where(object_pred2 > 0.45)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2 and subject[0] < len(mapping) \
                            and _start < len(mapping) and _end < len(mapping) and subject[1] < len(mapping) and _start != 0:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1,
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []


class SPO(tuple):
    def __init__(self, spo):
        self.spox = (
            tuple(spo[0]),
            spo[1],
            tuple(spo[2]),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def sequence_padding(inputs, length=None, padding=0):
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)


def evaluate(data, model, sess, tokenizer, max_len, id2predicate, limit=None):
    if limit:
        data = data[:limit]
    X, Y, Z = 1e-10, 1e-10, 1e-10
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in closed_single(d['text'], model, sess, tokenizer, max_len, id2predicate)])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        # pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))
        log.info('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))

    pbar.close()
    return f1, precision, recall

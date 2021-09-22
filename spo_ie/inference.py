# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : wangbingqian@boe.com.cn
@time   :20-11-27 上午11:42
@IDE    :PyCharm
@document   :inference2.py
"""

import numpy as np
import tensorflow
from spo_ie.tokenizers import Tokenizer
import json
import codecs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class IeModel(object):
    def __init__(self):
        # 启动会话
        self.max_len = 510
        self.tokenizer = OurTokenizer(self.load_vocab(os.path.join(os.path.dirname(__file__), '../config/vocab.txt')))
        self.id2predicate = self.load_schema(os.path.join(os.path.dirname(__file__), "../config/all_50_schemas_me.json"))
        self.schemas = self.load_org_schema(os.path.join(os.path.dirname(__file__), "../config/all_50_schemas"))

        self.model_name = os.path.join(os.path.dirname(__file__), "../model/train_model_v3_20201203.ckpt")
        session_config = tensorflow.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        self.__sess = tensorflow.Session(config=session_config)

        # 载入数据
        self.saver = tensorflow.train.import_meta_graph(self.model_name + '.meta')
        self.saver.restore(self.__sess, self.model_name)

        # 载入图结构
        self.__graph = tensorflow.get_default_graph()

    def extraction_spo(self, text):
        max_len = self.max_len
        token_ids, segment_ids = self.tokenizer.encode(text)
        input_mask = [1] * len(token_ids)
        token_ids = self.sequence_padding([token_ids])
        input_mask = self.sequence_padding([input_mask])
        segment_ids = self.sequence_padding([segment_ids])

        subject_preds1, subject_preds2 = self.__sess.run([self.__graph.get_tensor_by_name('s_dense/score_s1:0'),
                                                          self.__graph.get_tensor_by_name('s_dense/score_s2:0')],
                                                         feed_dict={'Inputs_id:0': token_ids,
                                                                    'Inputs_mask:0': input_mask,
                                                                    'segment_ids:0': segment_ids,
                                                                    'is_train:0': False
                                                                    })
        subject_preds1 = np.array(subject_preds1, dtype=np.float32)
        subject_preds2 = np.array(subject_preds2, dtype=np.float32)
        start = np.where(subject_preds1[0] > 0.5)[0]
        end = np.where(subject_preds2[0] > 0.5)[0]
        subjects = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))
        if subjects:
            spoes = []
            token_ids = np.repeat(token_ids, len(subjects), 0)
            input_mask = np.repeat(input_mask, len(subjects), 0)
            segment_ids = np.repeat(segment_ids, len(subjects), 0)
            subjects = np.array(subjects)
            object_preds1, object_preds2 = self.__sess.run([self.__graph.get_tensor_by_name('o_dense/score_o1:0'),
                                                            self.__graph.get_tensor_by_name('o_dense/score_o2:0')],
                                                           feed_dict={'Inputs_id:0': token_ids,
                                                                      'Inputs_mask:0': input_mask,
                                                                      'segment_ids:0': segment_ids,
                                                                      'tag_s_ids:0': subjects,
                                                                      'is_train:0': False
                                                                      })
            object_preds1 = np.array(object_preds1, dtype=np.float32)
            object_preds2 = np.array(object_preds2, dtype=np.float32)
            for subject, object_pred1, object_pred2 in zip(subjects, object_preds1, object_preds2):
                start = np.where(object_pred1 > 0.5)
                end = np.where(object_pred2 > 0.5)
                for _start, predicate1 in zip(*start):
                    for _end, predicate2 in zip(*end):
                        if _start <= _end and predicate1 == predicate2:
                            spoes.append((text[subject[0] - 1:subject[1]], predicate1, text[_start - 1:_end]))
                            break
            return list(set([(s, self.id2predicate[p], o) for s, p, o, in spoes]))
        else:
            return []

    @staticmethod
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

    @staticmethod
    def load_schema(schema_path):
        id2predicate, predicate2id = json.load(open(schema_path, encoding='utf-8'))
        id2predicate = {int(i) - 1: j for i, j in id2predicate.items()}
        return id2predicate

    @staticmethod
    def load_vocab(dict_path):
        token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return token_dict

    @staticmethod
    def load_org_schema(org_schema_path):
        schemas = {}
        with open(org_schema_path, encoding="utf8") as f:
            for l in f:
                a = json.loads(l)
                schemas[a['predicate']] = (a['subject_type'], a['object_type'])
        return schemas

    def output(self, text):
        spo_list = self.extraction_spo(text)
        res = {'spo_list': []}
        for _spo in spo_list:
            spo_dic = {}
            types = self.schemas[_spo[1]]
            spo_dic['source'] = _spo[2]
            spo_dic['source_type'] = types[1]
            spo_dic['name'] = _spo[1]
            spo_dic['target'] = _spo[0]
            spo_dic['target_type'] = types[0]
            res['spo_list'].append(spo_dic)
        return res


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


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





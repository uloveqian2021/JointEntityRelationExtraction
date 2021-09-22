#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 转化为GPU调用
import tensorflow as tf
from spo_ie.layer_norm_ops import ConditionalLayerNorm
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
from spo_ie import modeling
from spo_ie import optimization
import json

"""
利用　ConditionalLayerNorm　实现宾语的抽取
"""


class bert_m(object):
    # bert层相当于提取上层特征

    def __init__(self, config_path, init_checkpoint, input_ids, input_mask, segment_ids, is_training):
        model = modeling.BertModel(config=modeling.BertConfig.from_json_file(config_path),
                                   is_training=is_training,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   token_type_ids=segment_ids,
                                   use_one_hot_embeddings=False,
                                   scope='bert')

        # layer_logits = []
        # for i, layer in enumerate(model.all_encoder_layers):
        #     layer_feature = tf.layers.dense(layer, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        #                                     name="layer_logit%d" % i)
        #     layer_feature = tf.nn.relu(layer_feature)  # TODO tanh
        #     layer_logits.append(layer_feature)
        # layer_logits = tf.concat(layer_logits, axis=2)  # 第三维度拼接
        # layer_dist = tf.nn.softmax(layer_logits)
        # seq_out = tf.concat([tf.expand_dims(x, axis=2) for x in model.all_encoder_layers], axis=2)
        # pooled_output = tf.matmul(tf.expand_dims(layer_dist, axis=2), seq_out)
        # pooled_output = tf.squeeze(pooled_output, axis=2)
        # pooled_layer = pooled_output
        # char_bert_outputs = pooled_laRERyer[:, 1: max_seq_length - 1, :]  # [batch_size, seq_length, embedding_size]
        # char_bert_outputs = pooled_layer

        self.tvars = tf.trainable_variables()
        (self.assignment_map, _) = modeling.get_assignment_map_from_checkpoint(self.tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, self.assignment_map)
        # ===================
        # 分类采用     model.get_pooled_output()
        # 序列标注采用  model.get_sequence_output()
        # 多层融合采用　pooled_output
        self.output_last2 = model.get_all_encoder_layers()[-2]
        self.output = model.get_sequence_output()


class model_fn(object):
    def __init__(self, config_path, init_checkpoint, num_labels, max_len,
                 relation_vocab,
                 learning_rate, num_train_steps,
                 num_warmup_steps, batch_size):
        self.num_labels = num_labels
        self.relation_vocab = relation_vocab
        self.initializer = initializers.xavier_initializer()
        # self.relation_hidden_size = relation_hidden_size
        self.batch_size = batch_size
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Inputs_id")
        # S的标签
        self.tag_s1_inputs = tf.placeholder(dtype=tf.float32, shape=[None, None], name="tag_label_s1")  # (32, 164)
        self.tag_s2_inputs = tf.placeholder(dtype=tf.float32, shape=[None, None], name="tag_label_s2")  # (32, 164)
        self.tag_s_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="tag_s_ids")  # (32, 2)
        self.mask_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Inputs_mask")
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_train')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
        param_config = json.load(open(config_path))
        self.cln = ConditionalLayerNorm(param_config['hidden_size'])
        # ================
        # O的标签
        self.tag_o1_inputs = tf.placeholder(tf.float32, shape=[None, None, None], name='tag_o_label')  # (32, 164, 49)
        self.tag_o2_inputs = tf.placeholder(tf.float32, shape=[None, None, None], name='tag_o_label')  # (32, 164, 49)

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.seqlen = tf.cast(length, tf.int32)
        bert = bert_m(config_path, init_checkpoint, self.char_inputs, self.mask_inputs, self.segment_ids,
                      self.is_training)

        self.encoder = bert.output  # TODO
        self.encoder2 = bert.output_last2  # TODO
        mask = tf.cast(self.mask_inputs, tf.float32)
        mask = tf.expand_dims(mask, 2)
        self.tag_s1 = tf.expand_dims(self.tag_s1_inputs, 2)
        self.tag_s2 = tf.expand_dims(self.tag_s2_inputs, 2)
        selection_loss_s1, selection_loss_s2 = self.trans_s()

        loss_s1 = tf.reduce_sum(selection_loss_s1 * mask) / tf.reduce_sum(mask)
        loss_s2 = tf.reduce_sum(selection_loss_s2 * mask) / tf.reduce_sum(mask)

        selection_loss_op1, selection_loss_op2 = self.trans_o(max_len)

        loss_op1 = tf.reduce_sum(selection_loss_op1, 2, keepdims=True)
        loss_op1 = tf.reduce_sum(loss_op1 * mask) / tf.reduce_sum(mask)

        loss_op2 = tf.reduce_sum(selection_loss_op2, 2, keepdims=True)
        loss_op2 = tf.reduce_sum(loss_op2 * mask) / tf.reduce_sum(mask)

        self.loss = loss_s1 + loss_s2 + loss_op1 + loss_op2
        print('self.loss', self.loss)
        self.train_op = optimization.create_optimizer(self.loss, learning_rate, num_train_steps, num_warmup_steps, False)

    def trans_s(self):
        with tf.variable_scope('s_dense'):
            # select_v = tf.layers.dense(inputs=self.encoder, units=relation_hidden_size, activation=tf.nn.tanh)
            self.logit_s1 = tf.layers.dense(inputs=self.encoder, units=1)
            self.logit_s2 = tf.layers.dense(inputs=self.encoder, units=1)
            print('self.logit', self.logit_s1)  # self.logit Tensor("s_dense/pow:0", shape=(?, ?, 1), dtype=float32)
            prob_s1 = tf.nn.sigmoid(self.logit_s1, name='score_s1')
            prob_s2 = tf.nn.sigmoid(self.logit_s2, name='score_s2')
            self.prob_s1 = prob_s1
            self.prob_s2 = prob_s2
            selection_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tag_s1, logits=self.logit_s1)
            selection_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tag_s2, logits=self.logit_s2)
        return selection_loss1, selection_loss2

    @staticmethod
    def gather_subject_two(output, subject_ids):
        """根据subject_ids从output中取出subject的向量表征
        """
        index_s = subject_ids[:, :1]  # s对应的向量
        index_e = subject_ids[:, 1:]
        start = tf.batch_gather(output, index_s)  # shape=(batch_size, 1, 768)
        end = tf.batch_gather(output, index_e)
        return start, end

    @staticmethod
    def extract_subject(output, inputs):
        """根据subject_ids从output中取出subject的向量表征
        """
        subject_ids = inputs
        subject_ids = tf.cast(subject_ids, 'int32')
        start = tf.batch_gather(output, subject_ids[:, :1])
        end = tf.batch_gather(output, subject_ids[:, 1:])
        subject = tf.concat([start, end], axis=2)
        return subject[:, 0]

    def trans_o(self, max_len=None):
        with tf.variable_scope('o_dense'):
            # 获取对应s,p对应的向量表示
            # 方法2
            # subject_s, subject_e = self.gather_subject_two(self.encoder, self.tag_s_ids)  # shape=(?, 1536)
            # output = tf.add(self.encoder, subject_feature)
            subject_feature = self.extract_subject(self.encoder, self.tag_s_ids)
            # 利用条件Layer Norm提取宾语和关系
            output = self.cln(self.encoder2, cond=subject_feature)

            self.logit_o1 = tf.layers.dense(inputs=output, units=len(self.relation_vocab))
            self.logit_o2 = tf.layers.dense(inputs=output, units=len(self.relation_vocab))
            # print('v', u)  # Tensor("o_dense/dense_2/Tanh:0", shape=(?, ?, 100), dtype=float32
            # self.logit_o = tf.reshape(u, [-1, max_len, len(self.relation_vocab), 2])
            prob_o1 = tf.nn.sigmoid(self.logit_o1, name='score_o1')
            prob_o2 = tf.nn.sigmoid(self.logit_o2, name='score_o2')
            print('self.logit_o1', self.logit_o1)
            self.prob_o1 = prob_o1
            self.prob_o2 = prob_o2
            selection_loss_o1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tag_o1_inputs, logits=self.logit_o1)
            selection_loss_o2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tag_o2_inputs, logits=self.logit_o2)
        return selection_loss_o1, selection_loss_o2

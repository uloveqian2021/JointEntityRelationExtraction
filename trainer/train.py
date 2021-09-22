#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
# import tokenization
from spo_ie import modeling
from spo_ie.utils import *
from spo_ie.model_spo import model_fn
# from model_spo_v2 import model_fn
from spo_ie.tokenizers import Tokenizer, load_vocab
from my_scp import *
import datetime
import logging as log

log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO)
log.info("start training!")

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
online, bert_type, batch_size, max_len, lr = True, 'robert_base', 32, 256, 2e-5
# online, bert_type, batch_size, max_len, lr = False, 'chinese_rbt3', 4, 128, 1e-4
debug = False

if online:
    pretrain_model_path = '/data/datasets/tmp_data/pretrain_model/'
else:
    pretrain_model_path = '/wang/pretrain_model/'
if bert_type == 'robert_large':
    bert_path = pretrain_model_path + 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/'
    # bert_path = pretrain_model_path + 'chinese_roberta_wwm_ext_L-12_H-768_A-12/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'robert_base':
    bert_path = pretrain_model_path + 'chinese_roberta_wwm_ext_L-12_H-768_A-12/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'user_base':
    bert_path = pretrain_model_path + 'user_base/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_uer_chinese.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'user_large':
    bert_path = pretrain_model_path + 'user_large/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_uer_24_chinese.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'simbert':
    bert_path = pretrain_model_path + 'chinese_simbert_L-12_H-768_A-12/'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    config_path = bert_path + 'bert_config.json'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'albert':
    bert_path = pretrain_model_path + 'albert_base/'
    config_path = bert_path + 'albert_config.json'
    checkpoint_path = bert_path + 'model.ckpt-best'
    dict_path = bert_path + 'vocab_chinese.txt'
elif bert_type == 'chinese_rbt3':
    bert_path = pretrain_model_path + 'chinese_rbt3_L-3_H-768_A-12/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'nezha_large':
    bert_path = pretrain_model_path + 'NEZHA-Large-WWM/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'model.ckpt-346400'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'nezha_base':
    bert_path = pretrain_model_path + 'NEZHA-Base/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'model.ckpt-900000'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'wonezha':
    bert_path = pretrain_model_path + 'chinese_wonezha_L-12_H-768_A-12/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'
elif bert_type == 'ernie':
    bert_path = pretrain_model_path + 'ernie-512/'
    config_path = bert_path + 'bert_config.json'
    checkpoint_path = bert_path + 'bert_model.ckpt'
    dict_path = bert_path + 'vocab.txt'
if bert_type in ['robert_large', 'robert_base', 'simbert', 'user_large', 'ernie', 'roberta', 'chinese_rbt3']:
    bert_type = 'bert'
elif bert_type in ['nezha_large', 'nezha_base', 'nezha_wwm']:
    bert_type = 'nezha'

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("mode", 'train', "The input datadir.", )
flags.DEFINE_string("data_dir", '../lic2019', "The input data dir. Should con ""for the task.")
flags.DEFINE_string("output_dir", './model', "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("bert_config_file", config_path, "The config json file corresponding to the pre-trained BERT model.")
flags.DEFINE_string("init_checkpoint", checkpoint_path, "Initial checkpoint  BERT model).")
flags.DEFINE_string("vocab_file", dict_path, "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text.")

flags.DEFINE_integer("max_seq_length", max_len, "The maximum total input sequence length after WordPiece tokenization.")
flags.DEFINE_integer("batch_size", batch_size, "Total batch size for training.")
flags.DEFINE_float("learning_rate", lr, "The initial learning rate for Adam.")
flags.DEFINE_integer("num_train_epochs", 30, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for. ""E.g., 0.1 = 10% of training.")


# 验证集评估
def evaluate_val(valid_data, model, sess, tokenizer, max_len, id2predicate, limit=None):
    f1, precision, recall = evaluate(valid_data, model, sess, tokenizer, max_len, id2predicate, limit)
    best_test_f1 = model.best_dev_f1.eval()
    if f1 > best_test_f1:
        tf.assign(model.best_dev_f1, f1).eval()  # 赋值操作  将f1值赋给model.best_dev_f1
        print('precision: %.5f, recall: %.5f ,f1: %.5f,' % (precision, recall, f1))
    test_f1 = model.best_dev_f1.eval()
    log.info('precision: %.5f, recall: %.5f ,f1: %.5f, best_f1:%.5f' % (precision, recall, f1, test_f1))
    return f1 > best_test_f1


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


def main(_):
    # token_dict = {}
    # with codecs.open(FLAGS.vocab_file, 'r', 'utf8') as reader:
    #     for line in reader:
    #         # token = line.strip()
    #         token = line.strip().split('\t')[0]
    #         token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(load_vocab(FLAGS.vocab_file), do_lower_case=True)

    id2predicate, predicate2id = json.load(open('lic2019/all_50_schemas_me.json', encoding='utf-8'))
    id2predicate = {int(i) - 1: j for i, j in id2predicate.items()}
    predicate2id = {i: int(j) - 1 for i, j in predicate2id.items()}
    train_data = load_data('lic2019/train_data.json', True, debug=debug)
    valid_data = load_data('lic2019/dev_data.json', False, debug=debug)

    train_D = data_generator(train_data, FLAGS.batch_size)
    train_examples = train_data
    num_train_steps = int(len(train_examples) / FLAGS.batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model = model_fn(config_path=FLAGS.bert_config_file,
                     init_checkpoint=FLAGS.init_checkpoint,
                     num_labels=len(predicate2id),
                     max_len=FLAGS.max_seq_length,
                     relation_vocab=predicate2id,
                     learning_rate=FLAGS.learning_rate,
                     num_train_steps=num_train_steps,
                     num_warmup_steps=num_warmup_steps,
                     batch_size=FLAGS.batch_size
                     )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 模型保存路径
    checkpoint_path = os.path.join('model', 'train_model_v3_{}.ckpt'.format(datetime.datetime.now().strftime('%Y%m%d')))
    checkpoint_path0 = os.path.join('model', 'train_model_v3_{}.ckpt.data-00000-of-00001'.format(datetime.datetime.now().strftime('%Y%m%d')))
    checkpoint_path1 = os.path.join('model', 'train_model_v3_{}.ckpt.index'.format(datetime.datetime.now().strftime('%Y%m%d')))
    checkpoint_path2 = os.path.join('model', 'train_model_v3_{}.ckpt.meta'.format(datetime.datetime.now().strftime('%Y%m%d')))
    # ===============================
    # 加载bert_config文件
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state('model')

        # ===============================
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('mode_path %s' % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        # ============================
        for j in range(FLAGS.num_train_epochs):  # 30
            print('j', j)
            eval_los = 0.0
            count = 0
            step = 0
            for (batch_text, batch_token_ids, batch_mask, batch_segment_ids,
                 batch_subject_label_s1, batch_subject_label_s2,
                 batch_subject_ids, batch_object_label_o1, batch_object_label_o2) \
                    in train_D.__iter__(random=True,
                                        predicate2id=predicate2id,
                                        max_len=FLAGS.max_seq_length,
                                        tokenizer=tokenizer):
                count = count + 1
                feed = {model.char_inputs: batch_token_ids,
                        model.mask_inputs: batch_mask,
                        model.segment_ids: batch_segment_ids,
                        model.tag_s1_inputs: batch_subject_label_s1,
                        model.tag_s2_inputs: batch_subject_label_s2,
                        model.tag_s_ids: batch_subject_ids,
                        model.tag_o1_inputs: batch_object_label_o1,
                        model.tag_o2_inputs: batch_object_label_o2,
                        model.is_training: True

                        }
                # print(sess.run(model.seqlen, feed))
                step = step + 1
                loss, _ = sess.run([model.loss, model.train_op], feed)

                eval_los = loss + eval_los
                los = eval_los / count
                if step % 40 == 0:
                    log.info('epoch:{}, step:{}, loss:{}'.format(j, step, los))
                    # print('step', step, 'loss', los)
                # if step % 5000 == 0:
                #     best = evaluate_val(valid_data, model, sess, tokenizer, FLAGS.max_seq_length, id2predicate, limit=10000)
                #     if best:
                #         saver.save(sess, checkpoint_path)

            best = evaluate_val(valid_data, model, sess, tokenizer, FLAGS.max_seq_length, id2predicate)
            if best:
                saver.save(sess, checkpoint_path)


if __name__ == "__main__":
    tf.app.run()
    upload_file(remote_path="/", file_path='model/')


import tensorflow as tf
import numpy as np
import os
import time
import datetime
from model.textcnn import TextCNN
import pandas as pd
from tensorflow.contrib import learn
from keras.preprocessing import sequence
from keras.models import Sequential
import jieba
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from model.data_helpers import *
# Parameters
# ==================================================


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 100)")
tf.flags.DEFINE_integer("max_len", 100, "max length (default: 100)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
data = pd.read_csv('../data/train_first.csv')
data = data.sample(frac=1).reset_index(drop=True)
# 划分x  y
X = data['Discuss']
Y = data['Score'][:,None]
# y ONEHOT
# lb = preprocessing.LabelBinarizer()
# lb.fit([1, 2, 3, 4, 5])
# Y = lb.transform(Y)
#加载停用词
def get_stopwords(path):
    return [line.strip() for line in open(path,'r',encoding='utf-8').readlines()]
#句子去停用词
def removestopwords(sentence):
        stopwords_list=get_stopwords('../data/stopwords.txt')
        outstr=[]
        for word in sentence:
            if not word in stopwords_list:
                if word!='\n' and word!='\t':
                     outstr.append(word)
        return outstr

#分词 并去掉停用词
def cut(sentence):
    return removestopwords(jieba.cut(sentence))
#分词后的word
sentences=[cut(x) for x in X]
import itertools
from collections import Counter
def get_vocab(sentences):
    counts = Counter(list(itertools.chain.from_iterable(sentences)))
    # 选择超过10次的value
    vocab_list = []
    for word in counts:
        if counts[word] >= 10:
            vocab_list.append(word)
    vocab = sorted(vocab_list)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab, vocab_to_int
# 获取字典  字典索引表
vocab, vocab_to_int = get_vocab(sentences)
def get_sentence2int(sentences,vocab_to_int):
    reviews_ints = []
    for each in sentences:
        int_eachsententce=[]
        for word in each:
            if word in vocab_to_int:
                int_eachsententce.append(vocab_to_int[word])
            else:
                int_eachsententce.append(0)
        reviews_ints.append(int_eachsententce)
    reviews_ints=sequence.pad_sequences(reviews_ints, maxlen=FLAGS.max_len)
    return reviews_ints
X=get_sentence2int(sentences,vocab_to_int)




# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=X.shape[1],
            num_classes=Y.shape[1],
            vocab_size=len(vocab)+1,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step,  loss = sess.run(
                [train_op, global_step, cnn.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))


        # Generate batches
        batches =batch_iter(
            list(zip(X, Y)), FLAGS.batch_size, FLAGS.num_epochs,shuffle=False)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
        saver = tf.train.Saver()
        saver.save(sess, "checkpoints/sentiment.ckpt")


        #predict
        predict_data=pd.read_csv('../data/predict_first.csv')
        predict_sententces=[cut(x) for x in predict_data['Discuss']]
        predict_X=get_sentence2int(predict_sententces,vocab_to_int)

        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint('checkpoints/')
        with tf.Session()as sess:
            # sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_file)
            predict_arr = []
            for ii in range(0, len(predict_X), 100):
                for p in sess.run(cnn.scores, feed_dict={cnn.input_x: predict_X[ii:ii + 100], cnn.dropout_keep_prob: 1.0}):
                    predict_arr.append(p)
            with open('predict.txt','w') as fwrite:
                for p in predict_arr:
                    fwrite.write('{}\n'.format(p))


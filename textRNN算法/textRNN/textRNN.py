# textRNN 算法类

import os
import tensorflow as tf
import numpy as np
import json
from sklearn import metrics

#from utils import * 
from textRNN.utils import *

if not os.path.exists('vocab'):
    os.makedirs('vocab')
VOCAB = os.path.join('vocab', 'vocabs.txt') # 词表
WORD2ID = os.path.join('vocab', 'word2id.json') # 词-id映射字典
CATE2ID = os.path.join('vocab', 'cate2id.json') # 类别-id映射字典
CATEGORY= os.path.join('vocab', 'category.txt') # 类别表

SAVE_DIR = 'checkpoints' # 模型保存路径
BEST = os.path.join(SAVE_DIR, 'best_validation') # 最佳模型结果


class TextRNN(object):
    """textRNN文本分类模型"""
    def __init__(self, n_class=10,embed_dim=64, seq_len=600,vocab_size=5000, rnn='gru',
                n_layer=2, hidden_dim=128, keep_prob=0.8, lr=1e-3, batch_size=128, epoch=10,
                verbose=100, save=10):

        self.n_class = n_class
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.rnn = rnn
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.keep_ratio = keep_prob
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose
        self.save = save

        tf.reset_default_graph() # 注意
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_len], name = 'input_x') # 输入节点[数字id序列]
        self.input_y = tf.placeholder(tf.float32, [None, self.n_class]) ######### 构建模型时需要知道多少类
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') 
        self.__build_model()

    def __build_model(self):
        """构建RNN模型"""
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.hidden_dim)
            #return tf.keras.layers.GRUCell(self.hidden_dim)

        def dropout():
            if self.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        with tf.device('/cpu:0'):
            """embedding层"""
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embed_dim]) # [5000,64]
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('rnn'):
            """rnn层"""
            cells = [dropout() for _ in range(self.n_layer)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True) # 构建多层循环神经网络

            outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs= embedding_inputs, dtype=tf.float32)
            # https://blog.csdn.net/qq_35203425/article/details/79572514
            last = outputs[:, -1, :] # 取最后一个时序输出作为结果
        # 下面基本和textCNN一样
        with tf.name_scope('fc'):
            """全连接层"""
            #fc = tf.layers.dense(last, self.hidden_dim, name='fc1')
            fc = tf.keras.layers.Dense(self.hidden_dim, name='fc1')(last)
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            #self.logits = tf.layers.dense(fc, self.n_class, name='fc2') # 输出
            self.logits = tf.keras.layers.Dense(self.n_class, name='fc2')(fc)

            self.y_pred_class = tf.argmax(tf.nn.softmax(self.logits), 1) # 每行最大的索引即预测类别

        with tf.name_scope('optimizer'):
            """损失函数和优化器"""
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_class) # tf.equal判断数组相等
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # tf.cast数据类型转换True-1,tf.mean()全部取均值

    def __evaluate(self,sess, x, y):
        """在数据集上评估准确率和损失"""
        rows = len(x)
        batch = batch_iter(x, y, 128)
        total_loss = 0.0
        total_acc = 0.0
        for x_batch, y_batch in batch:
            l = len(x_batch)
            feed_dict = {self.input_x:x_batch, self.input_y:y_batch, self.keep_prob:1.0}
            #print("测试：feed_dict:",feed_dict)
            loss, acc = sess.run([self.loss, self.acc], feed_dict=feed_dict)
            total_loss += loss*l
            total_acc += acc*l
        return total_loss/rows, total_acc/rows

    def train(self,train_file, valid_file):
        """训练模型"""
        # 配置tensorboard
        tensorboard_dir = 'tensorboard'
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        tf.summary.scalar('loss', self.loss) # 生成损失标量图
        tf.summary.scalar('accuracy',self.acc)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)  # 指定一个文件用来保存图
        # 配置模型保存
        saver = tf.train.Saver()
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        # 加载训练集数据，初始化：字符转数字
        texts, labels = build_vocab(train_file, VOCAB, self.vocab_size) # 构建字符表，输出本地VOCAB
        print("训练集加载成功,共 ",len(texts)," 条数据。")
        self.cates, self.cate2id = cate2id(labels,CATE2ID,CATEGORY) # 标签映射为数字，cates标签列表, cate2id标签-数字字典
        self.words, self.word2id = word2id(VOCAB,WORD2ID) # 所有字符映射为数字，字符列表，字符-数字字典
        print("字符-id映射字典加载成功！")
        # self.vocab_size = len(words)
        x_train, y_train = preprocess(texts, labels, self.word2id, self.cate2id, self.seq_len) # 预处理数据集：字符转数字
        print("训练集数据预处理完成！")
        texts_val, labels_val = read_file(valid_file) # 验证集
        x_valid, y_valid = preprocess(texts_val, labels_val, self.word2id, self.cate2id, self.seq_len)
        print("验证集加载和预处理完成，共",len(x_valid),"条数据。")
        # 创建会话并初始化变量
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        writer.add_graph(self.session.graph)
        # 开始训练
        print("开始训练模型！")
        total_batch = 0
        best_acc_val = 0.0
        last_improved = 0
        required_imporve = 100 # 当性能不再提升，提前停止
        flag = False
        for epoch in range(self.epoch): # 训练轮数
            print('Epoch:',epoch + 1)
            batch_train = batch_iter(x_train,y_train,self.batch_size) # 产生批次数据
            for x_batch, y_batch in batch_train:
                feed_dict = {self.input_x:x_batch, self.input_y:y_batch,self.keep_prob:self.keep_ratio}
                # 保存训练过程数据
                if total_batch % self.save == 0:
                    s = self.session.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, total_batch) #调用其add_summary方法将训练过程数据保存在filewriter指定的文件中
                # 在验证集上验证并输出日志
                if total_batch % self.verbose == 0:
                    feed_dict[self.keep_prob] = 1.0
                    loss_train, acc_train = self.session.run([self.loss, self.acc], feed_dict=feed_dict)
                    loss_val, acc_val = self.__evaluate(self.session, x_valid, y_valid) # 验证集上评估
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=self.session, save_path=BEST)
                        improved_str = "*" # 用于输出日志显示
                    else:
                        improved_str = ""
                    # 打印信息
                    msg = 'Iter:{0:>6}, Train loss:{1:>6.2},Train acc:{2:>7.2%},\
                           Val loss:{3:>6.2}, Val acc:{4:>7.2%}  {5}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val,improved_str))

                self.session.run(self.optimizer, feed_dict=feed_dict) # 训练优化
                total_batch += 1

                if total_batch - last_improved > required_imporve:
                    print("模型性能在验证集上不再提升，提前停止训练……")
                    flag = True
                    break
            if flag:
                break

    def load_model(self, model_path):
        """加载模型"""
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session,save_path=model_path) # 加载模型
        print("模型加载完成！")


    def test(self,test_file):
        """评估模型"""
        texts_test, labels_test = read_file(test_file) # 测试集集
        # 加载字符-id映射表，类别-id映射表
        word2id = load_dict(WORD2ID)
        cate2id = load_dict(CATE2ID)
        cates = load_cate(CATEGORY)
        x_test, y_test = preprocess(texts_test, labels_test, word2id, cate2id, self.seq_len)
        print("测试集加载和预处理完成，共 ",len(x_test)," 条数据。")

        loss_test, acc_test = self.__evaluate(self.session, x_test, y_test) # 在测试集上评估

        msg = 'Test loss: {0:>6.2}, Test acc: {1:>7.2%}'
        print(msg.format(loss_test, acc_test))

        batch_size = 128
        rows = len(x_test)
        num_batch = int((rows-1)/batch_size) + 1
        y_test_class = np.argmax(y_test,1)
        y_pred_class = np.zeros(shape=len(x_test), dtype=np.int32)
        for i in range(num_batch):
            start = i*batch_size
            end = min((i+1)*batch_size, rows)
            feed_dict = {self.input_x:x_test[start:end], self.keep_prob:1.0}
            y_pred_class[start:end] = self.session.run(self.y_pred_class, feed_dict=feed_dict)

        print("精确率   召回率   F1分数")
        print(metrics.classification_report(y_test_class, y_pred_class, target_names=cates))

        print("混淆矩阵：")
        print(metrics.confusion_matrix(y_test_class, y_pred_class))


    def predict(self,file):
        """在一批待分类文本上进行预测"""
        # 预处理文本
        texts = []
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    text = line.strip()
                    if text:
                        texts.append(list(text))  # 字符级CNN [['字','符','级']]
                except:
                    pass
        print("数据加载完成，共 ",len(texts),"条数据")
        # 加载映射字典和类别信息
        word2id = load_dict(WORD2ID)
        cates = load_cate(CATEGORY)
        print("字典加载完成！")

        text_id = []
        for i in range(len(texts)):
            text_id.append(list(word2id[x] for x in texts[i] if x in word2id)) # 字符转id.需转列表
        x_pad = kr.preprocessing.sequence.pad_sequences(text_id, self.seq_len)  # 转为等长的序列
        print("数据预处理完成！")

        # 分批次进行预测
        batch_size = 128
        rows = len(x_pad)
        num_batch = int((rows-1)/batch_size) + 1
        y_pred_class = np.zeros(shape=len(x_pad), dtype=np.int32) # 预测结果数组
        for i in range(num_batch):
            start = i*batch_size
            end = min((i+1)*batch_size, rows)
            feed_dict = {self.input_x:x_pad[start:end], self.keep_prob:1.0}
            y_pred_class[start:end] = self.session.run(self.y_pred_class, feed_dict=feed_dict)
        pred = [cates[i] for i in y_pred_class] # 数字转文字类别
        return pred

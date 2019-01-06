# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def dnn(train_features, train_labels, val_features, val_labels, test_features):
    
    
    batch_size = 512
    max_epochs = 25
    learning_rate = 0.001
    
    
    n = train_features.shape[0]
    k = train_features.shape[1]
    train = DataSet(train_features, train_labels)
    iters_in_epoch = np.floor(n / batch_size)
    num_iters = np.int(iters_in_epoch * max_epochs)
    
    
    L1, L2, L3, L4, L5  = 256, 128, 64, 32, 1
    
    
    X     = tf.placeholder(tf.float32, [None, k])
    Y_hat = tf.placeholder(tf.float32, [None, 1])
    
    W1 = tf.Variable(tf.truncated_normal([k, L1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([L1]))

    W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
    b2 = tf.Variable(tf.zeros([L2]))

    W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([L3]))

    W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
    b4 = tf.Variable(tf.zeros([L4]))

    W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=0.1))
    b5 = tf.Variable(tf.zeros([L5]))
    
    
    
    ###### model #####
    
    Y1 = tf.nn.relu(tf.matmul(X,  W1) + b1)
    Y1 = tf.nn.dropout(Y1, 1.0)                 #no dropout
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
    Y2 = tf.nn.dropout(Y2, 1.0)
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
    Y3 = tf.nn.dropout(Y3, 1.0)
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)
    Y4 = tf.nn.dropout(Y4, 1.0)
    Y  = tf.matmul(Y4, W5) + b5
        
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y_hat, Y))))
    update = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    
    with tf.Session() as sess:
        
        sess.run(init)
        epoch = 0
        val_minimum = np.infty
        early_stop = 0
        best_cv, best_ct, best_it = 0, 0, 0
        
        print("Will train until validation_1-rmse hasn't improved in 10 epochs")
        
        CT = []
        
        for i in range (num_iters):
            
            if (early_stop < 10):
        
                
            
                batch_X, batch_Y = train.next_batch(batch_size)
                batch_Y = np.reshape(batch_Y, (batch_size,1))
                _, cost_train = sess.run([update,cost], {X : batch_X, Y_hat : batch_Y})
            
                CT.append(cost_train)
                
                if i%iters_in_epoch == 0:
                
                    batch_X, batch_Y = val_features, val_labels
                    batch_Y = np.reshape(batch_Y, (batch_Y.shape[0],1))
                    cost_val = sess.run(cost, {X : batch_X, Y_hat : batch_Y})
                    
                    print('[{}] \t validation_0-rmse:{:.6f}    validaton_1-rmse:{:.6f}'.format(epoch, np.mean(CT), cost_val))
                    
                    if cost_val < val_minimum :
                        val_minimum = cost_val
                        saver.save(sess, "/tmp/model.ckpt")
                        best_cv = cost_val
                        best_ct = np.mean(CT)
                        best_it = epoch
                        early_stop = 0
                    else:
                        early_stop += 1

                    epoch += 1
                    CT = []

                    
                
        print('Stopping. Best iteration:')
        print('[{}] \t validation_0-rmse:{:.6f}    validaton_1-rmse:{:.6f}'.format(best_it, best_ct, best_cv))
        saver.restore(sess, "/tmp/model.ckpt")
        
        batch_X  = test_features
        batch_Y  = np.arange(0, test_features.shape[0],1)
        batch_Y  = np.reshape(batch_Y, (batch_Y.shape[0],1))
        
        y_pre = sess.run([Y], {X : batch_X, Y_hat : batch_Y})
                
    
    return y_pre
                
                
            
                
                
            
    
                       
                       
                       
    

    
    
    
    
    
    
    
class DataSet:

    def __init__(self,features,labels):
        self._index_in_epoch = 0                            
        self._epochs_completed = 0
        self._features = features
        self._labels = labels
        self._num_examples = features.shape[0]
        pass

    @property
    def features(self):
        return self._features
    
    @property
    def labels(self):
        return self._labels

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)          
            np.random.shuffle(idx)                          
            self._features = self.features[idx]   
            self._labels   = self.labels[idx]                  

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            features_rest_part = self.features[start:self._num_examples]
            labels_rest_part = self.labels[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)         
            np.random.shuffle(idx0)                        
            self._features = self.features[idx0] 
            self._labels  = self.labels[idx0]                   

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples               
            end =  self._index_in_epoch  
            features_new_part =  self._features[start:end]  
            labels_new_part  =  self._labels [start:end] 
            return np.concatenate((features_rest_part, features_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._labels[start:end]
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf

from Net import nets_util as util
from Net.layers import (weight_variable, weight_variable_devonc, bias_variable, 
                            conv2d, deconv2d, max_pool, crop_and_concat)


class Trainer(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    
    """
    
    verification_batch_size = 4
    
    def __init__(self, net, batch_size=1, norm_grads=False, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        
    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate, 
                                                        staircase=True)
            
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost, 
                                                                                global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                     global_step=global_step)
        
        return optimizer
        
    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0)
        
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]))
        
        if self.net.summaries and self.norm_grads:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        # tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        # tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()        
        init = tf.global_variables_initializer()
        
        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)
        
        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)
        
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
        
        return init
    def _update_avg_gradients(self, avg_gradients, gradients, step):
        if avg_gradients is None:
            avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
        for i in range(len(gradients)):
            avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step+1)))) + (gradients[i] / (step+1))
            
        return avg_gradients



    def train(self, data_provider, output_path, training_iters=300, epochs=10, dropout=0.75, display_step=1, restore=False, write_graph=False, prediction_path = 'prediction'):
        """
        Lauches the training process
        
        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path
        
        init = self._initialize(training_iters, output_path, restore, prediction_path)
        
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)
            
            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
            
            # test_x, test_y = data_provider(self.verification_batch_size)
            # pred_shape = self.store_prediction(sess, test_x, test_y, "_init")
            
            # summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")
            
            avg_gradients = None
            for epoch in range(epochs):
                # total_loss = 0
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    batch_x, batch_y = data_provider(self.batch_size)
                     
                    # Run optimization op (backprop)
                    _, loss, lr, gradients = sess.run((self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node), 
                                                      feed_dict={self.net.x: batch_x,
                                                                 self.net.y: batch_y, #util.crop_to_shape(batch_y, pred_shape),
                                                                 self.net.keep_prob: dropout})
                    logging.info("Step: {:}, Loss: {:.4f}".format(step, loss))
                #     if self.net.summaries and self.norm_grads:
                #         avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
                #         norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                #         self.norm_gradients_node.assign(norm_gradients).eval()
                    
                #     if step % display_step == 0:
                #         self.output_minibatch_stats(sess, summary_writer, step, batch_x, batch_y) #util.crop_to_shape(batch_y, pred_shape))
                        
                #     total_loss += loss

                # self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                # self.store_prediction(sess, test_x, test_y, "epoch_%s"%epoch)
                    
                save_path = self.net.save(sess, save_path)
            logging.info("Optimization Finished!")
            
            return save_path
    
    # def error_rate(self, predictions, labels):
    #     """
    #     Return the error rate based on dense predictions and 1-hot labels.
    #     """
        
    #     return 100.0 - (
    #         100.0 *
    #         np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
    #         (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))

    # def store_prediction(self, sess, batch_x, batch_y, name):
    #     prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x, 
    #                                                          self.net.y: batch_y, 
    #                                                          self.net.keep_prob: 1.})
    #     pred_shape = prediction.shape
        
    #     loss = sess.run(self.net.cost, feed_dict={self.net.x: batch_x, 
    #                                                    self.net.y: util.crop_to_shape(batch_y, pred_shape), 
    #                                                    self.net.keep_prob: 1.})
        
    #     logging.info("loss= {:.4f}".format(loss))
              
    #     img = util.combine_img_prediction(batch_x, batch_y, prediction)
    #     util.save_image(img, "%s/%s.jpg"%(self.prediction_path, name))
        
    #     return pred_shape
    
    # def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
    #     logging.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))
    
    # def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
    #     # Calculate batch loss and accuracy
    #     summary_str, loss, predictions = sess.run([self.summary_op, 
    #                                                         self.net.cost, 
    #                                                         # self.net.accuracy, 
    #                                                         self.net.predicter
    #                                                         ], 
    #                                                        feed_dict={self.net.x: batch_x,
    #                                                                   self.net.y: batch_y,
    #                                                                   self.net.keep_prob: 1.})
    #     summary_writer.add_summary(summary_str, step)
    #     summary_writer.flush()
    #     logging.info("Iter {:}, Minibatch Loss= {:.4f}".format(step, loss))
   

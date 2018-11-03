# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import math
import pickle 

import tensorflow as tf
import numpy as np

import cifar10

slim = tf.contrib.slim
Normal = tf.contrib.distributions.Normal

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('init_dir', 'events/cifar10_train',
                           """Directory where to restore """
                           """from the checkpoint.""")
tf.app.flags.DEFINE_string('train_dir', 'events_GANT/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
#tf.app.flags.DEFINE_integer('max_steps', 1000000,
#                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_steps', 150000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")                             

def train(T_est):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step() 
    
    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      #images, labels = cifar10.distorted_inputs()
      #images, labels = cifar10.noisy_distorted_inputs()
      images, labels, T_tru, T_mask = cifar10.noisy_distorted_inputs(return_T_flag=True)
    
    T_est = tf.constant(T_est,dtype=tf.float32)
 
    #### Prior and groudtruth
    T_est = tf.tile(tf.expand_dims(T_est, 0),[FLAGS.batch_size,1,1])
    T_tru = tf.tile(tf.expand_dims(T_tru, 0),[FLAGS.batch_size,1,1])
    T_mask= tf.tile(tf.expand_dims(T_mask,0),[FLAGS.batch_size,1,1])

    #### generator
    with tf.variable_scope('generator') as scope:
       normal = Normal(tf.zeros([1,10]),tf.ones([1,10]))
       epsilon = tf.to_float(normal.sample(FLAGS.batch_size))
       net = slim.stack(epsilon,slim.fully_connected,[50,50])
       net = slim.fully_connected(net,cifar10.NUM_CLASSES*cifar10.NUM_CLASSES,activation_fn=None)
       net = tf.reshape(net,[-1,cifar10.NUM_CLASSES,cifar10.NUM_CLASSES])
    S = tf.nn.softmax(net)

    # input to discriminator
    S_mask = tf.sigmoid((S-0.05)/0.005)
    
    #### discriminator
    def discriminator(input):
       with tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE) as scope:
          input = slim.flatten(input)
          net = slim.fully_connected(input,20,activation_fn=tf.nn.sigmoid)
          net = slim.fully_connected(net,1,activation_fn=None)
       return net
    D_t = discriminator(T_mask)    
    D_s = discriminator(S_mask)
    
    #### reconstructor
    dropout = tf.constant(0.75)
    logits = cifar10.inference(images,dropout,dropout_flag=True)
    preds = tf.nn.softmax(logits)
    preds_aug = tf.reshape(tf.matmul(tf.reshape(preds,[FLAGS.batch_size,1,-1]),S),[FLAGS.batch_size,-1])
    logits_aug = tf.log(tf.clip_by_value(preds_aug,1e-8,1.0-1e-8))
 
    #### loss 
    # R loss   
    R_loss = cifar10.loss(logits_aug,labels)
    tf.summary.scalar('reconstructor loss',R_loss)

    # D loss
    D_loss = -tf.reduce_mean(D_t) + tf.reduce_mean(D_s)
    tf.summary.scalar('discriminator loss',D_loss)

    # G loss
    G_loss = R_loss - tf.reduce_mean(D_s)
    #G_loss = - tf.reduce_mean(D_s)
    tf.summary.scalar('generator loss',G_loss)
 
    # initialization of G 
    S_logits = tf.log(tf.clip_by_value(S,1e-8,1.0-1e-8))
    Initial_G_loss = -tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(T_est*S_logits,axis=2),axis=1))
    
    # variable list
    var_C = []
    var_D = []
    var_G = []
    for item in tf.trainable_variables():
       if "generator" in item.name:
          var_G.append(item)
       elif "discriminator" in item.name:
          var_D.append(item)
       else:
          var_C.append(item)

    #### optimizer
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    R_train_op, variable_averages, lr = cifar10.train(R_loss,global_step,var_C,return_variable_averages=True,return_lr=True)     
    lr_DG = tf.constant(1e-5) 
    D_train_op = tf.train.RMSPropOptimizer(learning_rate=lr_DG).minimize(D_loss,var_list=var_D)
    G_train_op = tf.train.RMSPropOptimizer(learning_rate=lr_DG).minimize(G_loss,var_list=var_G+var_C)

    #### optimizer for the initialization of the generator and the discriminator
    Initial_G_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(Initial_G_loss,var_list=var_G)

    #### weight clamping for WGAN
    clip_D = [var.assign(tf.clip_by_value(var,-0.01,0.005)) for var in var_D]

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._my_print_flag = False
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(R_loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          if self._my_print_flag:
             format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                           'sec/batch)')
             print (format_str % (datetime.now(), self._step, loss_value,
                                  examples_per_sec, sec_per_batch))

    #### build scalffold for MonitoredTrainingSession to restore the variables you wish
    ckpt = tf.train.get_checkpoint_state(FLAGS.init_dir)
    variables_to_restore = variable_averages.variables_to_restore()
    #print(variables_to_restore)
    for var_name in variables_to_restore.keys():
       if ('generator' in var_name) or ('discriminator' in var_name) or ('RMSProp' in var_name) or ('global_step' in var_name):
          del variables_to_restore[var_name]
    print(variables_to_restore)

    init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
         ckpt.model_checkpoint_path, variables_to_restore)
    def InitAssignFn(scaffold,sess):
       sess.run(init_assign_op, init_feed_dict)

    scaffold = tf.train.Scaffold(saver=tf.train.Saver(), init_fn=InitAssignFn)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    loggerHook = _LoggerHook()
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir = FLAGS.train_dir,
        scaffold = scaffold,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(R_loss),
               loggerHook],
        save_checkpoint_secs=60,
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)) as mon_sess:

      #### pretrain the generator
      loggerHook._my_print_flag = False
      res = None
      for i in xrange(10000):
         res = mon_sess.run([Initial_G_train_op,Initial_G_loss,T_est,S,lr,lr_DG])
         if i % 1000 == 0:
            print('Step: %d\tGenerator loss: %.3f'%(i,res[1]))
            print('Pre-estimation', res[2][0])
            print('Initialization', res[3][0])
            
      #### iteratively train G and <D,R>
      loggerHook._my_print_flag = False
      step = 0
      step_control = 0
      lr_, lr_DG_ = res[-2], res[-1]
      while not mon_sess.should_stop():
        # update the learning_rate of generator and discriminator to sync with the classifier
        if lr_DG_ >= lr_:  # to avoid over-tuning the transition matrix due to the learning_rate decay
           lr_DG_ = lr_DG_/10.0
        # do the adversarial game
        if step >= step_control:
           res = mon_sess.run([G_train_op,G_loss,T_est,S,T_tru,S_mask,T_mask],feed_dict={lr_DG:lr_DG_})  
           g_loss = res[1]
 
           for i in xrange(5):
              _, d_loss = mon_sess.run([D_train_op,D_loss],feed_dict={lr_DG:lr_DG_})

        # train the classifier
        _, r_loss, g_step, lr_, lr_DG_ = mon_sess.run([R_train_op,R_loss,global_step,lr,lr_DG],feed_dict={lr_DG:lr_DG_}) 
         
        if step >= step_control:  
           print('Step: %d\tR_loss: %.3f\tD_loss: %.3f\tG_loss: %.3f' % (g_step, r_loss, d_loss, g_loss))
        
           if (g_step % 2000 == 0) or (g_step == FLAGS.max_steps-1):
             print('Pre-estimation', res[2][0])
             print('Generated sample', res[3][0])
             print('True transition', res[4][0])
             print('Generated structure',res[5][0])
             print('True structure',res[6][0])
        else:  
           print('Step: %d\tR_loss: %.3f' % (g_step, r_loss))

        step = g_step

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  with open('T.pkl') as f:
     T = pickle.load(f) 
  print('estimated confusion matrix\n',T)
  train(T)

if __name__ == '__main__':
  tf.app.run()

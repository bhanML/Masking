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

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('init_dir','events/cifar10_train',
                           """Directory where to load the intializing weights""")
tf.app.flags.DEFINE_string('train_dir', 'events_varT/cifar10_train',
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
   
def estimation_T(anchor_set=True):
  """Estimation T based on a pretrained model"""
  with tf.Graph().as_default():
    images, labels = cifar10.inputs(eval_data=False)   # on the training data
    logits = cifar10.inference(images)
    pred = tf.nn.softmax(logits)
    
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    with tf.Session() as sess:
	  ckpt = tf.train.get_checkpoint_state(FLAGS.init_dir)
	  if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
	  else:
		print('No checkpoint file found')
		return
	
	  # start the queue runner
	  coord = tf.train.Coordinator()
	  try:
		threads = []
		for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
			threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))  
		num_iter = int(math.ceil(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size))
		step = 0
		preds = []
                annotations = []
		while step < num_iter:
			#print('step: ', step)
			res = sess.run([pred,labels])
			preds.append(res[0])
                        annotations.append(res[1])
			step += 1
			
	  except Exception as e:
		coord.request_stop(e)
		  
	  coord.request_stop()
	  coord.join(threads, stop_grace_period_secs=10)
    
  preds = np.concatenate(preds,axis=0)
  #print(preds.shape)
  annotations = np.concatenate(annotations,axis=0)
  if anchor_set:
     indices = np.argmax(preds,axis=0)
     #print(indices)
     est_T = np.array(np.take(preds,indices,axis=0))
  else:
     unnormal_est_T = np.zeros((cifar10.NUM_CLASSES,cifar10.NUM_CLASSES))
     for i in xrange(annotations.shape[0]):
       label = annotations[i]
       unnormal_est_T[:,label] += preds[i]
     unnormal_est_T_sum = np.sum(unnormal_est_T,axis=1)
     est_T = unnormal_est_T / unnormal_est_T_sum[:,None]
  return est_T

def train(T_est):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
   
    logits_T_init = np.log(T_est + 1e-8) 
    logits_T = tf.get_variable('logits_T',shape=[cifar10.NUM_CLASSES,cifar10.NUM_CLASSES],initializer=tf.constant_initializer(logits_T_init))
    T = tf.nn.softmax(logits_T)

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      #images, labels = cifar10.distorted_inputs()
      images, labels, T_tru, T_mask_tru = cifar10.noisy_distorted_inputs(return_T_flag=True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    dropout = tf.constant(0.75)
    logits = cifar10.inference(images,dropout,dropout_flag=True)

    # softmax layer 
    preds =tf.nn.softmax(logits)
    preds_aug = tf.clip_by_value(tf.matmul(preds,T), 1e-8, 1.0 - 1e-8)
    logits_aug = tf.log(preds_aug)

    # Calculate loss.
    loss = cifar10.loss(logits_aug, labels)
    
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op, variable_averages = cifar10.train(loss, global_step, return_variable_averages=True)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    
    #### build scalffold for MonitoredTrainingSession to restore the variables you wish 
    ckpt = tf.train.get_checkpoint_state(FLAGS.init_dir)
    variables_to_restore = variable_averages.variables_to_restore()
    #print(variables_to_restore)
    for var_name in variables_to_restore.keys():
       if ('logits_T' in var_name) or ('global_step' in var_name):
          del variables_to_restore[var_name]
    #print(variables_to_restore)

    init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
         ckpt.model_checkpoint_path, variables_to_restore)
    def InitAssignFn(scaffold,sess):
       sess.run(init_assign_op, init_feed_dict)

    scaffold = tf.train.Scaffold(saver=tf.train.Saver(), init_fn=InitAssignFn)
 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir = FLAGS.train_dir,
        scaffold = scaffold,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        save_checkpoint_secs=60,
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)) as mon_sess:

      while not mon_sess.should_stop():
        res = mon_sess.run([train_op,global_step,T_tru,T_mask_tru])
        if res[1] % 1000 == 0:
          print('Disturbing matrix\n',res[2])
          print('Masked structure\n',res[3])


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  T = estimation_T()
  print('estimated T \n', T)
  with open('varT.pkl','w') as w:
     pickle.dump(T,w)
  train(T)


if __name__ == '__main__':
  tf.app.run()

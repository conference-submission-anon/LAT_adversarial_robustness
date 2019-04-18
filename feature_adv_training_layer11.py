import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from model import Model as OrigModel
from feature_model import FeatureModel as Model
import json
import math
import cifar10_input
import os
from pgd_attack import LinfPGDAttack
from datetime import datetime

with open('config_layer11.json') as config_file:
  config = json.load(config_file)

model_dir = config['model_dir']
new_model_dir = config['new_model_dir']
cur_checkpoint = tf.train.latest_checkpoint(model_dir)


if not os.path.exists(new_model_dir):
  os.makedirs(new_model_dir)

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']
raw_cifar = cifar10_input.CIFAR10Data(data_path)



def pgd_attack(im_nat, label, x_placeholder, y_placeholder, grad, s = 0.005, eps = 16.0/255.0, steps = 100, randomize = True) :

        min_val = np.min(im_nat)
        max_val = np.max(im_nat)
        eps = eps * (max_val - min_val)
        s = s * (max_val - min_val)
        # print(min_val, max_val, eps, s)

        if randomize :      
            im = im_nat + np.random.uniform(-eps, eps, im_nat.shape)
        else :
            im = np.array(im_nat)   
        for i in range(steps):
            g_ = sess.run(grad, feed_dict = {x_placeholder: im, y_placeholder : label})
            im += s * np.sign(g_)
            # if len(im.shape) == 2 :
            im = np.clip(im, im_nat - eps, im_nat + eps)
            im = np.clip(im, min_val, max_val)                
        return im

def modified_pgd_attack(im_nat, label, x_placeholder, y_placeholder, grad, s = 0.005, eps = 16.0/255.0, steps = 100, randomize = True) :

    min_val = np.reshape(np.min(im_nat, axis = (0, 1, 2)), [1, 1, 1, -1]) * np.ones(im_nat.shape)
    max_val = np.reshape(np.max(im_nat, axis = (0, 1, 2)), [1, 1, 1, -1]) * np.ones(im_nat.shape)
    eps = eps * (max_val - min_val)
    s = s * (max_val - min_val)

    if randomize :
      im = im_nat + np.random.uniform(-eps, eps, im_nat.shape)
    else :
      im = np.array(im_nat)
    for i in range(steps):
      g_ = sess.run(grad, feed_dict = {x_placeholder: im, y_placeholder : label})
      im += s * np.sign(g_)
      im = np.clip(im, im_nat - eps, im_nat + eps)
      im = np.clip(im, min_val, max_val)
    return im

model = OrigModel(mode = 'train')

attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'])

saver1 = tf.train.Saver()
total_vars1 = tf.global_variables()
train_vars1 = tf.trainable_variables()
model2 = Model(placeholder = 11, scope = 'New_model', mode = 'train')
train_vars2 = tf.trainable_variables()[len(train_vars1):]
total_vars2 = tf.global_variables()[len(total_vars1):]
model2_update_ops = [var2.assign(var1) for (var1, var2) in zip(train_vars1, train_vars2)]
model1_update_ops = [var1.assign(var2) for (var1, var2) in zip(train_vars1, train_vars2)]

initial_update_ops = [var2.assign(var1) for (var1, var2) in zip(total_vars1, total_vars2)]
model2_grad = tf.gradients(model2.mean_xent, model2.x_placeholder)[0]

model1_train_step = tf.train.MomentumOptimizer(1e-3, momentum).minimize(model.mean_xent + weight_decay * model.weight_decay_loss)
model2_train_step = tf.train.MomentumOptimizer(2.5e-4, momentum).minimize(model2.mean_xent)

saver2 = tf.train.Saver(train_vars2)

sess = tf.InteractiveSession()
cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)
sess.run(tf.global_variables_initializer())
saver1.restore(sess, cur_checkpoint)

s = 1.0/255.0
eps = 8.0/255.0
steps = 20

sess.run(initial_update_ops)
max_num_training_steps = config['max_num_training_steps']

for ii in range(max_num_training_steps):
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                       multiple_passes=True)
    sess.run(model2_update_ops)
    new_x_batch = sess.run(model2.out, feed_dict = {model2.x_input : x_batch})
    new_x_batch_adv = modified_pgd_attack(new_x_batch, y_batch, model2.x_placeholder, model2.y_input,  model2_grad,
                        s, eps, steps)

    sess.run(model2_train_step, feed_dict = {model2.x_placeholder : new_x_batch_adv, model2.y_input : y_batch})

    sess.run(model1_update_ops)
    x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    adv_dict = {model.x_input: x_batch_adv,
                model.y_input: y_batch}

    sess.run(model1_train_step, feed_dict=adv_dict)

    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict={model.x_input : x_batch, model.y_input : y_batch})
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))

    if ii % num_checkpoint_steps == 0 and ii > 0 :
        saver1.save(sess, os.path.join(new_model_dir, 'checkpoint'), global_step=ii)
        print('model saved')

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from model import Model as OrigModel
from feature_model import FeatureModel as Model
import json
import math
import cifar10_input
import os
from adversarial_generation import LinfPGDAttack
from datetime import datetime

with open('config_perturbation.json') as config_file:
  config = json.load(config_file)

model_dir = config['model_dir']
new_model_dir = config['new_model_dir']
cur_checkpoint = tf.train.latest_checkpoint(model_dir)

if not os.path.exists(new_model_dir):
  os.makedirs(new_model_dir)

data_path = config['data_path']
batch_size = config['eval_batch_size']
raw_cifar = cifar10_input.CIFAR10Data(data_path)

model = OrigModel(mode='eval')

saver1 = tf.train.Saver()
train_vars1 = tf.trainable_variables()
model1_vars = tf.global_variables()
layer_target = config['layer_to_attack']
model2 = Model(placeholder = layer_target, scope = 'New_model', mode = 'eval')

attack = LinfPGDAttack(model2, 
                       config['epsilon'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])
total_vars = tf.global_variables()

train_vars2 = tf.global_variables()[len(total_vars)/2:]
model2_update_ops = [var2.assign(var1) for (var1, var2) in zip(model1_vars, train_vars2)]

model_grad = tf.gradients(model.xent,model.x_input)[0]
model2_grad = tf.gradients(model2.mean_xent, model2.x_placeholder)[0]

saver2 = tf.train.Saver(train_vars2)

sess = tf.InteractiveSession()
cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver1.restore(sess,cur_checkpoint)
    sess.run(model2_update_ops)

    x_batch, y_batch = cifar.eval_data.get_next_batch(batch_size,
                                                       multiple_passes=True)
    acc = sess.run(model.accuracy, feed_dict = {model.x_input : x_batch, model.y_input : y_batch})
    
    print('The orginal model batch accuracy is ',acc)

    out = sess.run(model2.out, feed_dict = {model2.x_input : x_batch})
    acc = sess.run(model2.accuracy, feed_dict = {model2.x_placeholder : out, model2.y_input : y_batch})
    print('The model2 batch accuracy is ', acc)

    training_time = 0.0
    total_test_size = config['total_test_size']
    no_of_iter = int(np.ceil(total_test_size/np.float32(batch_size)))
    adv_adv_acc = 0.0
    adv_acc_iter = 0.0
    adv_acc = 0.0
    pgd_with_adv_count = 0.0
    pgd_with_adv_iter_count = 0.0
    adv_images_iter = []
    adv_labels_iter = []
    
    for i in range(np.int(no_of_iter)):
        bstart = i * batch_size
        bend = min(bstart + batch_size, total_test_size)

        x_batch = raw_cifar.eval_data.xs[bstart:bend, :]
        y_batch = raw_cifar.eval_data.ys[bstart:bend]

        # Compute Adversarial Perturbations
        d_eps = 0
        correct_adv_iter, x_adv_iter_to_save = attack.LA_perturb(model,x_batch,y_batch,d_eps,sess)

        nat_dict = {model.x_input: x_batch,
                  model.y_input: y_batch}

        adv_acc_iter_temp = (np.sum(correct_adv_iter)/(np.float32(batch_size)))
        print("advance adv iter accuracy : ",adv_acc_iter_temp," for batch in ",i)

        adv_acc_iter += adv_acc_iter_temp
        print("agg adv iter acc: ", adv_acc_iter/np.float32(i+1))

        adv_labels_iter.append(np.array(correct_adv_iter))
        adv_images_iter.append(x_adv_iter_to_save)
    
    adv_acc_iter /= no_of_iter


    if (config['l2_activate'] == True):
        print("l2 attack ", "layer no ", layer_target)
    else:
        print("l_inf attack ", "layer no ", layer_target)
    print('*********************')
    print('    LA attack accuracy {:.4}%'.format(adv_acc_iter * 100))
    print('*********************')

    adv_labels_iter = np.concatenate(adv_labels_iter)
    adv_images_iter = np.concatenate(adv_images_iter)   
    adv_examples_dir = config['store_adv_path']
    if not os.path.isdir(adv_examples_dir):
            os.mkdir(adv_examples_dir)

    if (config['l2_activate'] == True):
        np.save(str(adv_examples_dir)+"/labels_l2_layer_"+str(layer_target)+".npy", (adv_labels_iter))
        np.save(str(adv_examples_dir)+"/images_l2_layer_"+str(layer_target)+".npy", (adv_images_iter))
    else:
        np.save(str(adv_examples_dir)+"/labels_linf_layer_"+str(layer_target)+".npy", (adv_labels_iter))
        np.save(str(adv_examples_dir)+"/images_linf_layer_"+str(layer_target)+".npy", (adv_images_iter))

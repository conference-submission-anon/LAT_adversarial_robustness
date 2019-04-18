from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import cifar10_input
import json

with open('config_perturbation.json') as config_file:
    config = json.load(config_file)

class LinfPGDAttack:
    def __init__(self, model, epsilon, step_size, random_start, loss_func):
        """Attack parameter initialization. The attack performs k steps of
             size a, while always staying within epsilon from the initial
             point."""
        self.model = model
        self.epsilon = epsilon
        # self.k = num_steps
        self.a = step_size
        self.eps_l2 = config['eps_l2']
        self.rand = random_start
        self.k_latent_h = config['k_latent_h']
        self.k_latent_x = config['k_latent_x']
        self.k_joint_iter = config['k_joint_iter']
        self.a_latent = config['a_latent']
        self.x_lat_adv = tf.placeholder(shape = self.model.out.get_shape().as_list(), dtype = tf.float32)
        if loss_func == 'xent':
            loss = model.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(model.y_input,                   10,
                                                            on_value=1.0,
                                                            off_value=0.0,
                                                            dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

        self.grad = tf.gradients(loss, model.x_input)[0]
        self.grad_latent = tf.gradients(loss, model.x_placeholder)[0]
        self.grad_feature = tf.gradients(tf.abs(self.x_lat_adv-self.model.out), self.model.x_input)[0]
        self.grad_cosine = tf.gradients(tf.losses.cosine_distance(self.x_lat_adv, self.model.out, dim=1), self.model.x_input)[0]
        self.grad_l2 = tf.gradients(tf.norm(self.x_lat_adv-self.model.out), self.model.x_input)[0]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
             examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, 0, 255) # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        for i in range(self.num_steps):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                                                        self.model.y_input: y})

            x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 255) # ensure valid pixel range
        return x


    def LA_perturb(self, original_model, x_nat, y, d_eps, sess):

        x = np.copy(x_nat)
        x_adv_iter_to_save = np.copy(x_nat)
        flag = np.zeros(len(y))
        
        correct_adv_iter = sess.run(original_model.correct_prediction, feed_dict=
                {original_model.x_input: x_nat, original_model.y_input: y})

        for j in range(self.k_joint_iter):
            x = np.array(x)
            x_lat = sess.run(self.model.out, feed_dict={self.model.x_input: x,
                                                self.model.y_input: y})

            axis_shape = tuple(range(len(x_lat.shape)-1))
            reshape_value = (np.repeat(1,len(x_lat.shape)))
            reshape_value[-1] = -1
            reshape_value = tuple(reshape_value)
            eps_max = 1.1*np.ones(x_lat.shape)*((np.max(x_lat,axis=axis_shape).reshape(reshape_value)))
            eps_min = 1.0*np.ones(x_lat.shape)*((np.min(x_lat,axis=axis_shape).reshape(reshape_value)))
            a_l = self.a_latent

            x_lat_adv = np.copy(x_lat)
            if (config['l2_activate'] == True):
                ##############
                # l_2 attack #
                ##############
                eps_l2 = self.eps_l2
                for i in range(self.k_latent_h):  
                    grad = sess.run(self.grad_latent, feed_dict={self.model.x_placeholder: x_lat_adv,
                                                        self.model.y_input: y,
                                                        })
                    grad = grad / np.linalg.norm(grad)
                    x_lat_adv += grad * a_l
                    x_lat_adv = np.clip(x_lat_adv, eps_min, eps_max)

                    delta = x_lat_adv - x_lat
                    n = np.linalg.norm(delta)
                    if n > eps_l2:
                        x_lat_adv = x_lat + (delta)/n * eps_l2

            else:
                ################
                # l_inf attack #
                ################
                for i in range(self.k_latent_h):
                    grad = sess.run(self.grad_latent, feed_dict={self.model.x_placeholder: x_lat_adv,
                    self.model.y_input: y,
                    })

                    x_lat_adv += a_l * np.sign(grad)
                    x_lat_adv = np.clip(x_lat_adv, eps_min, eps_max) 

            if j==1:
                print("latent layer robustness: ",sess.run(self.model.accuracy,feed_dict = {self.model.x_placeholder: x_lat_adv,
                  self.model.y_input: y}))

            for i in range(self.k_latent_x):
                grad = sess.run(self.grad_feature, feed_dict={self.model.x_input: x, self.x_lat_adv : x_lat_adv})                             
                
                if i==0:
                    m = np.zeros(grad.shape)

                m = 1.0*m + grad/((np.sum(abs(grad),axis=(1,2,3)))[:,None,None,None])
                x = np.subtract(x, self.a*np.sign(m), out=x, casting='unsafe')
                x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) # ensure eps bound of perturbed image
                x = np.clip(x, 0, 255) # ensure valid pixel range
            
            correct_adv_iter_batch = sess.run(original_model.correct_prediction, feed_dict=
              {original_model.x_input: x, original_model.y_input: y})
            correct_adv_iter = np.logical_and(correct_adv_iter_batch, correct_adv_iter)

            for i in range(len(y)):
                if(flag[i]==0 and correct_adv_iter_batch[i]==0):
                  flag[i]=1
                  x_adv_iter_to_save[i] = np.array(x[i])

        return correct_adv_iter, (x_adv_iter_to_save)


import tensorflow as tf
import variables
import tflearn
import numpy as np
import matplotlib.pyplot as plt

class Conscience:
    def __init__(self, sess, num_actor_vars):
        self.sess = sess
        self.learning_rate = 0.001
        self.tau = 0.001

        # Create the critic network
        self.input_states, self.input_action, self.output_score = self.create_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.output_score_reflex = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.output_score_reflex, self.output_score)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.output_score, self.input_action)

    def create_network(self):
        input_states = tflearn.input_data(shape=[None, variables.state_shape])
        input_action = tflearn.input_data(shape=[None, variables.action_shape])
        net = tflearn.fully_connected(input_states, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(input_action, 300)

        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(input_action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return input_states, input_action, out

    def train(self, state, action, output_score_reflex):
        return self.sess.run([self.output_score, self.optimize], feed_dict={
            self.input_states: state,
            self.input_action: action,
            self.output_score_reflex: output_score_reflex
        })

    def predict(self, state, action):
        return self.sess.run(self.output_score, feed_dict={
            self.input_states: state,
            self.input_action: action
        })

    def action_gradients(self, state, action):
        return self.sess.run(self.action_grads, feed_dict={
            self.input_states: state,
            self.input_action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

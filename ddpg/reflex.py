import tensorflow as tf
import tflearn
import variables

class Reflex:
    def __init__(self, sess):
        self.sess = sess

        self.learning_rate = 0.0001
        self.tau = 0.001
        self.batch_size = variables.batch_size

        self.input_states, self.output_action = self.create_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_input_states,  self.target_out = self.create_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, variables.action_shape])

        # Combine the gradients here

        self.unnormalized_actor_gradients = tf.gradients(self.output_action, self.network_params, -self.action_gradient)
        print(self.unnormalized_actor_gradients)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_network(self):
        input_states = tflearn.input_data(shape=[None, variables.state_shape])
        net = tflearn.fully_connected(input_states, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, variables.action_shape, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, variables.action_max)
        return input_states, scaled_out

    def train(self, input_states, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.input_states: input_states,
            self.action_gradient: a_gradient
        })

    def predict(self, input_states):
        return self.sess.run(self.output_action, feed_dict={
            self.input_states: input_states
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_input_states: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

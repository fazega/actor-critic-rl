import tensorflow as tf
import variables

class Reflex:
    def __init__(self):
        self.learning_rate = 0.0001

        self.input_state = tf.placeholder(tf.float64, shape=(1,variables.state_shape))
        self.output_action_conscience = tf.placeholder(tf.float64, shape=(1,variables.action_shape))

        layer1 = tf.layers.dense(self.input_state, units=50, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(layer1, units=50, activation=tf.nn.leaky_relu)
        layer3 = tf.layers.dense(layer2, units=30, activation=tf.nn.leaky_relu)
        self.output_action = tf.layers.dense(layer3, units=variables.action_shape)

        loss = tf.losses.mean_squared_error(labels=self.output_action_conscience, predictions=self.output_action)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        compute_g = optimizer.compute_gradients(loss)
        self.apply_g = optimizer.apply_gradients(compute_g)

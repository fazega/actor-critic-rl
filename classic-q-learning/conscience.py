import tensorflow as tf
import variables
import numpy as np
import matplotlib.pyplot as plt

class Conscience:
    def __init__(self):
        self.learning_rate1 = 0.0001
        self.learning_rate2 = 0.1

        self.input_state = tf.placeholder(tf.float64, shape=(1,variables.state_shape))
        self.input_action = tf.Variable(np.zeros((1,variables.action_shape)))
        self.final_input = tf.concat([self.input_state, self.input_action], 1)

        self.layer1 = tf.layers.dense(self.final_input, units=50, activation=tf.nn.leaky_relu, name="layer1")
        self.layer2 = tf.layers.dense(self.layer1, units=70, activation=tf.nn.leaky_relu)
        # self.layer3 = tf.layers.dense(self.layer2, units=50, activation=tf.nn.softplus)
        self.layer4 = tf.layers.dense(self.layer2, units=30, activation=tf.nn.softplus)
        self.output_score = tf.layers.dense(self.layer4, units=1)
        self.output_score_reflex = tf.placeholder(tf.float64)

        loss = tf.losses.mean_squared_error(labels=self.output_score_reflex, predictions=self.output_score)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate1)
        l = [x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        l.remove(self.input_action)
        compute_g = optimizer.compute_gradients(loss, var_list=l)
        self.apply_g = optimizer.apply_gradients(compute_g)

        self.loss2 = tf.negative(self.output_score) #il faut maximiser le score
        for i in range(len(variables.action_domains)):
            (m,M) = variables.action_domains[i]
            self.loss2 *= ((tf.sign((self.input_action-m))+1)/2)
            self.loss2 *= ((-tf.sign(self.input_action-M)+1)/2)
            self.loss2 += ((-tf.sign((self.input_action-m))+1)/2)*(((self.input_action-m)**2+1)*tf.abs(self.output_score))
            self.loss2 += ((tf.sign((self.input_action-M))+1)/2)*(((self.input_action-M)**2+1)*tf.abs(self.output_score))
            # loss2 = loss2 + ((tf.sign(-(self.input_action-m))+1)/2 + (tf.sign(self.input_action-M)+1)/2)*(self.output_score)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate2)
        self.compute_g2 = optimizer2.compute_gradients(self.loss2, var_list=[self.input_action])
        self.reset_optimizer_best = tf.variables_initializer(optimizer2.variables())
        self.optimizer_best = optimizer2.apply_gradients(self.compute_g2)

    def find_best(self, session, state, activate=False):
        session.run(self.reset_optimizer_best)
        if(activate):
            print("Initial action : "+str(session.run(self.input_action, feed_dict={self.input_state:state})))
            l = []
            actions = []
            for i in range(100):
                action = -6 + i*(12/100)
                actions.append(action)
                self.input_action.load(np.matrix(action), session)
                l.append(session.run(self.loss2, feed_dict={self.input_state:state})[0,0])
        # self.input_action.load(np.matrix(action), session)
        for i in range(300):
            x = session.run(self.optimizer_best, feed_dict={self.input_state:state})
            if(activate):
                print(session.run(self.input_action, feed_dict={self.input_state:state}))
            #print(session.run(self.output_score, feed_dict={self.input_state:state}))
        # print(session.run(self.input_action, feed_dict={self.input_state:state}))
        if(activate):
            plt.plot(actions,l)
            plt.show()
        return session.run(self.input_action, feed_dict={self.input_state:state})

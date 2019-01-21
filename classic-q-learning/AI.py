import numpy
import tensorflow as tf
import time
import math

from conscience import *
from reflex import *
from threading import Thread,Condition

class AI:
    def __init__(self):
        self.conscience = Conscience()
        self.reflex = Reflex()
        self.epsilon_random = lambda t : 0.7*(1/(1+0.00002*t)) #proba de tirer une action random pour l'exploration
        self.epsilon_update_conscience = 0.3
        self.iteration_training = 20
        self.gamma = 0.9
        self.depth = 2
        self.depth_step = 4

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.data_in = []

        self.thread_signal = Condition()
        self.thread_train = Thread(target = self.updateNetworks)
        self.thread_train.start()

    def getAction(self,state, t):
        action = self.sess.run(self.reflex.output_action, feed_dict={self.reflex.input_state:state,})
        #action = np.array([int(round(x,0)) for x in action[0]])[0]
        if action < variables.action_domains[0][0]:
            action = np.matrix([[variables.action_domains[0][0]]])
        if action > variables.action_domains[0][1]:
            action = np.matrix([[variables.action_domains[0][1]]])
        if(numpy.random.rand() < self.epsilon_random(t)):
            return np.matrix(variables.action_space.sample()),True
        return action,False

    def updateNetworks(self):
        while True:
            self.thread_signal.acquire()
            while len(self.data_in) < 2*self.depth*self.depth_step:
                self.thread_signal.wait()

            for k in range(len(self.data_in)-self.depth*self.depth_step):
                trajectory = self.data_in[k:(k+self.depth*self.depth_step)][::self.depth_step]
                score = trajectory[0][2]
                # self.conscience.input_action.load(trajectory[0][1], self.sess)
                # Q_value = self.sess.run(self.conscience.output_score, feed_dict={self.conscience.input_state:trajectory[0][0]})
                for i in range(1,len(trajectory)):
                    elem = trajectory[i]
                    best_action_next = self.sess.run(self.reflex.output_action, feed_dict={self.reflex.input_state:elem[0],})
                    self.conscience.input_action.load(best_action_next, self.sess)
                    score += (self.gamma**i) * self.sess.run(self.conscience.output_score, feed_dict={self.conscience.input_state:elem[0]})
                # score = Q_value + (1/4)*(score - Q_value)
                # print("Score : "+str(score))
                (state,action,reward) = trajectory[0]
                self.conscience.input_action.load(action, self.sess)
                for i in range(self.iteration_training):
                    self.sess.run(self.conscience.apply_g, feed_dict={self.conscience.input_state:state, self.conscience.output_score_reflex:score})
                    # print(self.sess.run(self.conscience.output_score, feed_dict={self.conscience.input_state:state}))

                new_best_action = self.conscience.find_best(self.sess, state, activate=False)
                # print("Best action --> "+str(new_best_action))
                #Tous les N tours, on met à jour le réflexe (N=4 là)
                for i in range(self.iteration_training):
                    self.sess.run(self.reflex.apply_g, feed_dict={self.reflex.input_state:state, self.reflex.output_action_conscience: new_best_action})
            self.data_in = self.data_in[(len(self.data_in)-self.depth):-1]
            self.thread_signal.notify()
            self.thread_signal.release()

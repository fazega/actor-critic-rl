import numpy
import tensorflow as tf
import time
import math
import random

from conscience import *
from reflex import *
from threading import Thread,Condition

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class BufferExperience:
    def __init__(self):
        self.data = []

    def getBatch(self, batch_size):
        batch = random.sample(self.data, batch_size) # choose an episode
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        return s_batch, a_batch, r_batch, s2_batch, t_batch



class AI:
    def __init__(self):
        self.sess = tf.Session()

        self.reflex = Reflex(self.sess)
        self.conscience = Conscience(self.sess, self.reflex.num_trainable_vars)

        self.epsilon_random = lambda t : 0.6*(1/(1+0.0005*t)) #proba de tirer une action random pour l'exploration
        self.gamma = 0.99
        self.depth = 2
        self.depth_step = 1

        init = tf.global_variables_initializer()
        self.sess.run(init)


        self.buffer = BufferExperience()

        self.thread_signal = Condition()
        self.thread_train = Thread(target = self.updateNetworks)
        self.thread_train.start()

        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(variables.action_shape))

    def stateRepresentation(self,state,t):
        digits = t//2000
        return state.round(digits+2)

    def getAction(self, state, t):
        action = self.reflex.predict(state) + self.actor_noise()
        #action = np.array([int(round(x,0)) for x in action[0]])[0]
        # if(numpy.random.rand() < self.epsilon_random(t)):
        #     return np.matrix(variables.action_space.sample()),True
        return action,False

    import time
    def updateNetworks(self):
        while True:
            self.thread_signal.acquire()
            while len(self.buffer.data) < variables.batch_size:
                self.thread_signal.wait()
            batch = self.buffer.getBatch(variables.batch_size)
            # print(batch)
            # print("Learning")
            self.thread_signal.notify()
            self.thread_signal.release()

            (s_batch, a_batch, r_batch, s2_batch, t_batch) = self.buffer.getBatch(variables.batch_size)
            t0 = time.time()
            target_q = self.conscience.predict_target(s2_batch, self.reflex.predict_target(s2_batch))
            score_batch = []
            for i in range(len(target_q)):
                score = r_batch[i]
                if(not t_batch[i]):
                    score += self.gamma * target_q[i]
                score_batch.append(score)

            # print((s_batch,a_batch,np.reshape(score_batch, (len(score_batch),1))))
            self.conscience.train(s_batch, a_batch, np.reshape(score_batch, (len(score_batch),1)))
            a_outs = self.reflex.predict(s_batch)
            grads = self.conscience.action_gradients(s_batch, a_outs)
            self.reflex.train(s_batch, grads[0])

            self.conscience.update_target_network()
            self.reflex.update_target_network()

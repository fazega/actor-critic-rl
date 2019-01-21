import numpy as np
import time
import math
import gym
import time
from AI import AI
import tensorflow as tf
import variables
import threading

delta_t = 0.1

ai = AI()

env = gym.make('Pendulum-v0')
t = 0
for i_episode in range(10000000):
    state = np.transpose(env.reset())
    variables.action_space = env.action_space;
    time0 = time.time()
    #writer = tf.summary.FileWriter("output", sess.graph)
    data_in = []
    while True:
        env.render()
        #On recupere ce que donne le réflexe
        action,isActionRandom = ai.getAction(np.matrix(state),t)
        print("Time "+str(t))
        str_random = "random" if isActionRandom else "NOT random"
        # print("State "+str(state)+" --> action "+str(action)+"   ("+str_random+")")
        newState, reward, done, info = env.step(action)
        reward = reward.item()
        data_in.append((np.matrix(state),np.matrix(action),reward))
        if(ai.thread_signal.acquire(blocking=False)):
            ai.data_in += data_in.copy()
            data_in = []
            ai.thread_signal.notify()
            ai.thread_signal.release()
        state = np.transpose(newState)
        # print("Reward : "+str(reward))
        if done:
            break
        #On met à jour la conscience, donc descente de gradient pour mieux prédire les scores
        #Input_action dans la conscience est en fait une variable, donc il faut la loader avec celle qu'on veut
        #variable car optimization par la suite pour trouver le max
        # if(len(trajectory)>ai.depth):
        #     ai.updateNetworks(trajectory[-ai.depth:],t)
        if(t%5000 == 0):
            ai.depth = ai.depth +0
        t+=1
    # writer.close()
    print("Episode : "+str(i_episode))

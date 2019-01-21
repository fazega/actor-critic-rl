import numpy as np
import time
import math
import gym
import time
from AI import AI
import tensorflow as tf
import variables
import threading
import matplotlib.pyplot as plt

delta_t = 0.1

ai = AI()

env = gym.make('Pendulum-v0')
t = 0
scores = []
for i_episode in range(50):
    print("Episode : "+str(i_episode)+ " -- Time : "+str(t))
    state = np.reshape(env.reset(), (variables.state_shape,))
    variables.action_space = env.action_space;
    time0 = time.time()
    #writer = tf.summary.FileWriter("output", sess.graph)
    mean_score_episode = 0
    count_iterations = 0

    while True:
        count_iterations += 1
        t+=1
        env.render()
        #On recupere ce que donne le réflexe
        action,isActionRandom = ai.getAction(ai.stateRepresentation(np.reshape(state, (1,variables.state_shape)),t),t)
        # print("Time "+str(t))
        str_random = "random" if isActionRandom else "NOT random"
        # print("State "+str(state)+" --> action "+str(action)+"   ("+str_random+")")
        newState, reward, done, info = env.step(action)
        reward = reward.item()
        mean_score_episode += reward
        # if(ai.thread_signal.acquire(blocking=False)):
        if(ai.thread_signal.acquire()):
            ai.buffer.data.append((ai.stateRepresentation(state,t),np.reshape(action, (variables.action_shape,)),reward,ai.stateRepresentation(np.reshape(newState, (variables.state_shape,)),t),done))
            ai.thread_signal.notify()
            ai.thread_signal.release()
        state = np.reshape(newState, (variables.state_shape,))
        # print("Reward : "+str(reward))
        if done:
            break
        #On met à jour la conscience, donc descente de gradient pour mieux prédire les scores
        #Input_action dans la conscience est en fait une variable, donc il faut la loader avec celle qu'on veut
        #variable car optimization par la suite pour trouver le max
        # if(len(trajectory)>ai.depth):
        #     ai.updateNetworks(trajectory[-ai.depth:],t)
        # if(t%5000 == 0):
        #     ai.depth = ai.depth +0
        if(count_iterations == 99):
            mean_score_episode /= count_iterations
            scores.append(mean_score_episode)
            mean_score_episode = 0
            count_iterations = 0
print(scores)
plt.plot(scores)
plt.show()


    # writer.close()

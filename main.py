
# -*- coding: utf-8 -*-

from World import World
import numpy as np


if __name__ == "__main__":

    env = World()
    env.reset()
    # The Optimal Values by Dynamic Programming
    dp=[0,0.285,0.076,0.007,0.747,0.576,0,-0.086,0.928,0.584,0.188,0.08,0,0,0,-0.086]
    MSE_list=np.zeros((3,9))
    QMSE=np.zeros(9)
    total_episodes=100000
    alphas=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    decay_rates=[0.9995,0.99995,0.999995]

    ##SARSA Tuning
    # indexe = 0
    # for dr in decay_rates:
    #     indexa = 0
    #     for alpha in alphas:
    #         Q=env.sarsa(total_episodes,alpha,dr)
    #         policy=np.zeros(env.get_nstates())
    #         value = np.zeros(env.get_nstates())
    #         for i in range(env.get_nstates()):
    #             policy[i]=(np.argmax(Q[i,:])+1)
    #             value[i] = (np.max(Q[i, :]))
    #         # env.plot_policy(policy)
    #         # env.plot_actionValues(Q)
    #         summation = 0  # variable to store the summation of differences
    #         n = len(policy)  # finding total number of items in list
    #         for i in range(0, n):  # looping through each element of the list
    #             difference = dp[i] - value[i]  # finding the difference between observed and predicted value
    #             squared_difference = difference ** 2  # taking square of the differene
    #             summation = summation + squared_difference  # taking a sum of all the differences
    #         MSE = summation / n  # dividing summation by total values to obtain average
    #         print(MSE)
    #         MSE_list[indexe,indexa]=MSE
    #         indexa+=1
    #     indexe += 1

    ##Qlearning Tuning
    # indexa = 0
    # for alpha in alphas:
    #     Q=env.Qlearning(total_episodes,alpha)
    #     policy=np.zeros(env.get_nstates())
    #     value = np.zeros(env.get_nstates())
    #     for i in range(env.get_nstates()):
    #         policy[i]=(np.argmax(Q[i,:])+1)
    #         value[i] = (np.max(Q[i, :]))
    #     # env.plot_policy(policy)
    #     # env.plot_actionValues(Q)
    #     summation = 0  # variable to store the summation of differences
    #     n = len(policy)  # finding total number of items in list
    #     for i in range(0, n):  # looping through each element of the list
    #         difference = dp[i] - value[i]  # finding the difference between observed and predicted value
    #         squared_difference = difference ** 2  # taking square of the differene
    #         summation = summation + squared_difference  # taking a sum of all the differences
    #     MSE = summation / n  # dividing summation by total values to obtain average
    #     print(MSE)
    #     QMSE[indexa]=MSE
    #     indexa+=1


    # print("final MSE SARSA")
    # for i in range(0,3):
    #     for j in range(0,9):
    #         print(MSE_list[i,j])
    #     print("___________")


    # print("final MSE Qlearning")
    # for j in range(0,9):
    #    print(QMSE[j])


    Q = env.sarsa(total_episodes, 0.01, 0.99995)
    policy = np.zeros(env.get_nstates())
    value = np.zeros(env.get_nstates())
    for i in range(env.get_nstates()):
        policy[i] = (np.argmax(Q[i, :]) + 1)
        value[i] = (np.max(Q[i, :]))
    env.plot_value(value)
    env.plot_policy(policy)
    env.plot_actionValues(Q)

    Q = env.Qlearning(total_episodes, 0.01)
    policy = np.zeros(env.get_nstates())
    value = np.zeros(env.get_nstates())
    for i in range(env.get_nstates()):
        policy[i] = (np.argmax(Q[i, :]) + 1)
        value[i] = (np.max(Q[i, :]))
    env.plot_value(value)
    env.plot_policy(policy)
    env.plot_actionValues(Q)



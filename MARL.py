import numpy as np
from GridWorld import GridWorld, Action
import matplotlib.pyplot as plt
from itertools import product
from Utils import *

def shareStateQLearning (env, stop, seed) :
    gen = np.random.RandomState(seed)

    # Each predator has own action.
    qList = [dict() for _ in range(env.nPredator)]

    # Initialize qList  
    for i, Q in enumerate(qList) :
        Q[i] = gen.rand(4)
        xRange = range(-env.rows, env.rows + 1)
        yRange = range(-env.cols, env.cols + 1)
        for ps in product(xRange, yRange) :
            if ps == (0, 0) :
                Q[ps] = np.zeros(4)
            else :
                Q[ps] = gen.rand(4)

    timestep = 0
    episode = 0

    ts = []
    es = []

    while stop(episode):
        # Initialization of predator states
        predatorStates = [i for i in range(env.nPredator)]

        while not env.jointTerminate():

            aList = []

            for i in range(env.nPredator) :
                Q = qList[i]
                s = predatorStates[i]
                a = env.selectAction(Q, s, (T / np.log2(episode + 3)))
                aList.append(a)

            # Move the prey.
            env.movePrey()

            for i in range(env.nPredator) :
                Q = qList[i]
                s = predatorStates[i]
                a = aList[i]
                # Get next state and reward
                s_, r = env.next(i, s, a)

                # Update this predator's Q function.
                Q[s][a.value] += BETA * (r + GAMMA * np.max(Q[s_]) - Q[s][a.value])
                # Set up state and action for next iteration
                predatorStates[i] = s_
                # Broadcast state to fellow predators.
                env.broadcast(i)

            # Increment the timestep
            timestep += 1

        # Re-initialize the environment at the
        # end of the episode
        env.initialize()
        episode += 1


        es.append(episode)
        ts.append(timestep)

    return qList, es, ts

def independentQLearning (env, stop, seed) :
    gen = np.random.RandomState(seed)

    # Independent Q-Learning where each
    # predator has its own action-value function.
    qList = [dict() for _ in range(env.nPredator)]

    # Initialize qList  
    for i, Q in enumerate(qList) :
        Q[i] = gen.rand(4)
        perceptRange = range(-env.perceptWindow, env.perceptWindow + 1)
        for ps in product(perceptRange, perceptRange) :
            if ps == (0, 0) :
                Q[ps] = np.zeros(4)
            else :
                Q[ps] = gen.rand(4)

    timestep = 0
    episode = 0

    ts = []
    es = []

    while stop(episode):
        # Initialization of predator states
        predatorStates = [i for i in range(env.nPredator)]

        while not env.jointTerminate():

            aList = []

            for i in range(env.nPredator) :
                Q = qList[i]
                s = predatorStates[i]
                a = env.selectAction(Q, s, (T / np.log2(episode + 3)))
                aList.append(a)

            # Move the prey.
            env.movePrey()

            for i in range(env.nPredator) :
                Q = qList[i]
                s = predatorStates[i]
                a = aList[i]
                # Get next state and reward
                s_, r = env.next(i, s, a)

                # Update this predator's Q function.
                Q[s][a.value] += BETA * (r + GAMMA * np.max(Q[s_]) - Q[s][a.value])
                # Set up state and action for next iteration
                predatorStates[i] = s_

            # Increment the timestep
            timestep += 1

        # Re-initialize the environment at the
        # end of the episode
        env.initialize()
        episode += 1

        es.append(episode)
        ts.append(timestep)

    return qList, es, ts



def main () :
    env = GridWorld()
    qList, es1, ts1 = independentQLearning(env, lambda x : x < 100, 0)
    _, es2, ts2 = shareStateQLearning(env, lambda x : x < 100, 0)
    iQL  = plt.scatter(es1, ts1, c='red')
    ssQL = plt.scatter(es2, ts2, c='blue')
    iQL.set_label("Independent")
    ssQL.set_label("3 Predators, 1 Prey, Share State")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative TimeSteps")
    plt.legend()
    plt.show()
    env.simulateTrajectory(qList)

if __name__ == "__main__" : 
    main()

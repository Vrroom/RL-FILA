import numpy as np
from GridWorld import GridWorld, Action
import matplotlib.pyplot as plt
from itertools import product
from Utils import *

def independentQLearning (env, stop, seed, plot=False) :
    # Independent Q-Learning where each
    # predator has its own action-value function.
    gen = np.random.RandomState(seed)

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
        preyCaught = False

        while not preyCaught:

            aList = []

            for i in range(env.nPredator) :
                Q = qList[i]
                s = predatorStates[i]
                a = env.selectAction(Q, s, (T / (episode + 1)))
                aList.append(a)

            # Move the prey.
            env.movePrey()

            for i in range(env.nPredator) :
                Q = qList[i]
                s = predatorStates[i]
                a = aList[i]
                # Get next state and reward
                s_, r = env.next(i, s, a)
                # Check whether prey has been caught.
                if s_ == (0, 0) :
                    preyCaught = True

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

    if plot :
        plt.plot(es, ts)
        plt.show()

    return qList

def main () :
    env = GridWorld()
    qList = independentQLearning(env, lambda x : x < 100, 0, plot=True)
    env.simulateTrajectory(qList)

if __name__ == "__main__" : 
    main()

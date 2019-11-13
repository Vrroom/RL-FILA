import numpy as np
from GridWorld import GridWorld, Action
import matplotlib.pyplot as plt
from Utils import *

def independentQLearning (env, stop, seed, plot=False) :
    # Independent Q-Learning where each
    # predator has its own action-value function.
    gen = np.random.RandomState(seed)

    qList = [dict() for _ in range(env.nPredator)]
    timestep = 0
    episode = 0

    ts = []
    es = []

    while stop(episode):
        # Initialization of predator states
        predatorStates = [i for i in range(env.nPredator)]

        for i in range(env.nPredator) :
            s = predatorStates[i]
            Q = qList[i]
            if not s in Q :
                Q[s] = gen.rand(4)

        # The epsilon-t for epsilon-t greedy
        epsilon = 0.2 / (episode + 1)

        preyCaught = False

        while not preyCaught:

            for i in range(env.nPredator) :
                Q = qList[i]
                s = predatorStates[i]
                a = env.selectAction(Q, s, T)
                # Get next state and reward
                s_, r = env.next(i, s, a) 
                # Check whether prey has been caught.
                if s_ == (0, 0) :
                    preyCaught = True
                if not s_ in Q :
                    # If s_ is the goal state
                    # for any predator, we 
                    # initialize Q[s_] = [0, 0, 0, 0] 
                    # as written in Sutton & Barto. 
                    # Else we initialize randomly.
                    if s_ == (0, 0) : 
                        Q[s_] = np.zeros(4)
                    else :
                        Q[s_] = gen.rand(4)

                # Update this predator's Q function.
                Q[s][a.value] += BETA * (r + GAMMA * np.max(Q[s_]) - Q[s][a.value])
                # Set up state and action for next iteration
                predatorStates[i] = s_

            # Increment the timestep and move the prey
            timestep += 1
            env.movePrey()

        # Re-initialize the environment at the
        # end of the episode
        env.initialize()
        episode += 1


        es.append(episode)
        ts.append(timestep)
        print(episode, timestep)

    if plot :
        plt.plot(ts, es)
        plt.show()

    return qList

def main () :
    env = GridWorld()
    qList = independentQLearning(env, lambda x : x < 100, 0, plot=True)
    env.simulateTrajectory(qList)

if __name__ == "__main__" : 
    main()
import numpy as np
from GridWorld import GridWorld, Action
import matplotlib.pyplot as plt
from itertools import product
from Utils import *

def shareStateQLearning (env, stop, seed) :
    gen = np.random.RandomState(seed)
    nActions = len(env.actions)
    # Each predator has own action.
    qList = [dict() for _ in range(env.nPredator)]

    # Initialize qList  
    for i, Q in enumerate(qList) :
        Q[i] = gen.rand(nActions)
        xRange = range(-env.rows, env.rows + 1)
        yRange = range(-env.cols, env.cols + 1)
        for ps in product(xRange, yRange) :
            if ps == (0, 0) :
                Q[ps] = np.zeros(nActions)
            else :
                Q[ps] = gen.rand(nActions)

    timestep = 0
    episode = 0

    ts = []
    es = []

    while stop(episode):
        # Initialization of predator states
        predatorStates = [i for i in range(env.nPredator)]

        while not env.terminate():

            aList = []
            # Selecting action for each predator
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
        if(episode%100==0):
            last_t = 0 if len(ts)==0 else ts[-1]
            print("Shared episode - %d\tepisode length = %d\taverage steps/eps = %d"%(episode,timestep-last_t,timestep//episode))
        es.append(episode)
        ts.append(timestep)
    print("End")
    return qList, es, ts

def independentQLearning (env, stop, seed) :
    gen = np.random.RandomState(seed)
    nActions = len(env.actions)
    # Independent Q-Learning where each
    # predator has its own action-value function.
    qList = [dict() for _ in range(env.nPredator)]

    # Initialize qList  
    for i, Q in enumerate(qList) :
        Q[i] = gen.rand(nActions)
        perceptRange = range(-env.perceptWindow, env.perceptWindow + 1)
        for ps in product(perceptRange, perceptRange) :
            if ps == (0, 0) :
                Q[ps] = np.zeros(nActions)
            else :
                Q[ps] = gen.rand(nActions)

    timestep = 0
    episode = 0

    ts = []
    es = []

    while stop(episode):
        # Initialization of predator states
        predatorStates = [i for i in range(env.nPredator)]

        while not env.terminate():

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
        if(episode%100==0):
            last_t = 0 if len(ts)==0 else ts[-1]
            print("Indi episode - %d\tepisode length = %d\taverage steps/eps = %d"%(episode,timestep - last_t,timestep//episode))
        es.append(episode)
        ts.append(timestep)
    print("End")
    return qList, es, ts

def QLearning (env, stop, seed, sharing=False) :
    gen = np.random.RandomState(seed)
    nActions = len(env.actions)
    # Each predator has own action.
    qList = [(dict(),dict()) for _ in range(env.nPredator)]

    # Initialize qList  
    for i, (Q,QC) in enumerate(qList) :
        Q[i] = np.zeros(nActions)
        QC[i] = np.zeros(nActions)
        xRange = range(-env.rows, env.rows + 1)
        yRange = range(-env.cols, env.cols + 1)
        for ps in product(xRange, yRange) :
            QC[ps]=np.zeros(nActions)
            if ps == (0, 0):
                Q[ps] = np.zeros(nActions)
            else :
                Q[ps] = np.zeros(nActions)

    timestep = 0
    episode = 0

    ts = []
    es = []
    avg=[]
    while stop(episode):
        # Initialization of predator states
        predatorStates = [i for i in range(env.nPredator)]

        while not env.terminate():

            aList = []
            # Selecting action for each predator
            for i in range(env.nPredator) :
                Q,_ = qList[i]
                s = predatorStates[i]
                a = env.selectAction(Q, s, (T / np.log2(episode + 3)))
                aList.append(a)

            # Move the prey.
            env.movePrey()

            for i in range(env.nPredator) :
                Q,QC = qList[i]
                s = predatorStates[i]
                a = aList[i]
                # Get next state and reward
                s_, r = env.next(i, s, a)
                # print(s,s_)
                # Update this predator's Q function.
                Q[s][a.value] += BETA * (r + GAMMA * np.max(Q[s_]) - Q[s][a.value])
                QC[s][a.value] +=1
                # Set up state and action for next iteration
                predatorStates[i] = s_
                # Broadcast state to fellow predators.
                # if(sharing):
                #     env.broadcast(i)

            # Increment the timestep
            timestep += 1

        # Re-initialize the environment at the
        # end of the episode
        env.initialize()
        episode += 1
        if(episode%100==0):
            last_t = 0 if len(ts)==0 else ts[-1]
            # print("episode %d\r"%(episode),end="")
            print("%d episode - %d\tepisode length = %d\taverage steps/eps = %d\taverage last 50 eps = %d"%(int(sharing),episode,timestep-last_t,timestep//episode,(timestep-ts[-50])//50))
        es.append(episode)
        ts.append(timestep)
        avg.append(timestep//episode)
    print("End")
    return qList, es, ts, avg

def Test_run(env, stop, qList, seed, sharing=False):
    gen = np.random.RandomState(seed)
    nActions = len(env.actions)

    timestep = 0
    episode = 0

    bcount = 0
    totalcount = 0

    ts = []
    es = []
    avg=[]
    while stop(episode):
        # Initialization of predator states
        predatorStates = [i for i in range(env.nPredator)]

        while not env.terminate():

            aList = []
            # Selecting action for each predator
            for i in range(env.nPredator) :
                Q,_ = qList[i]
                s = predatorStates[i]
                a = env.selectAction(Q, s, (T / np.log2(episode + 3)))
                aList.append(a)

                if a == Action.Stay:
                    bcount = bcount + 1
                totalcount = totalcount + 1

            # Move the prey.
            env.movePrey()

            for i in range(env.nPredator) :
                Q,_ = qList[i]
                s = predatorStates[i]
                a = aList[i]
                # Get next state and reward
                s_, r = env.next(i, s, a)

                # Update this predator's Q function.
                # Q[s][a.value] += BETA * (r + GAMMA * np.max(Q[s_]) - Q[s][a.value])
                # Set up state and action for next iteration
                predatorStates[i] = s_
                # Broadcast state to fellow predators.
                # if(sharing):
                #     env.broadcast(i)

            # Increment the timestep
            timestep += 1

        # Re-initialize the environment at the
        # end of the episode
        env.initialize()
        episode += 1
        if(episode%100==0):
            last_t = 0 if len(ts)==0 else ts[-1]
            print("episode %d\r"%(episode),end="")
            # print("%d episode - %d\tepisode length = %d\taverage steps/eps = %d\taverage last 50 eps = %d"%(int(sharing),episode,timestep-last_t,timestep//episode,(timestep-ts[-50])//50))
        es.append(episode)
        ts.append(timestep)
        avg.append(1.0*timestep/episode)
    print("End")

    print(bcount)
    print(totalcount)
    return es, ts, avg
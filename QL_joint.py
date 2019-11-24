import numpy as np
from GridWorld import GridWorld, Action
import matplotlib.pyplot as plt
from itertools import product
from Utils import *

def QLearning_joint(env, stop, seed,aware=False,sharing=False,pretrain=False,_qList=None):

    gen = np.random.RandomState(seed)
    nActions = len(env.actions)
    # Each predator has own action.
    qList = [(dict()) for _ in range(env.nPredator)]

    # Initialize qList  
    for i, (Q) in enumerate(qList):
        Q[i] = np.zeros(nActions)
        xRange = range(-env.rows, env.rows + 1)
        yRange = range(-env.cols, env.cols + 1)
        for ps in product(xRange, yRange):
            Q[(i,ps)] = np.zeros(nActions)
            Q[ps] = np.zeros(nActions)
        # QC[i] = np.zeros(nActions)
        for ps in product(list(product(xRange, yRange)),list(product(xRange, yRange))):
            Q[ps] = np.zeros(nActions)
    if pretrain:
        qList=_qList
    timestep = 0
    episode = 0

    ts = []
    es = []
    avg=[]
    while stop(episode):
        # Initialization of predator states
        predatorStates = [i for i in range(env.nPredator)]
        print(episode)
        while not env.jointTerminate():
            # print(timestep)
            aList = []
            # Selecting action for each predator
            for i in range(env.nPredator):
                Q = qList[i]
                s = predatorStates[i]
                # if(aware and isinstance(s,tuple) and False):
                #     if(episode<TIED_PARAMS):
                #         # print("share")
                #         a = env.selectAction(Q, s[0], (T / 100.0))
                #     else:
                #         a = env.selectAction(Q, s, (T / 100.0))
                # else:
                a = env.selectAction(Q, s, (T / np.log2(episode + 3)))
                aList.append(a)

            # Move the prey.
            env.movePrey()
            env.movePredator(aList)
            for i in range(env.nPredator) :
                Q = qList[i]
                s = predatorStates[i]
                a = aList[i]
                # Get next state and reward
                if(aware):
                    s_, r = env.joint_next(i, s, a)
                else:
                    s_, r = env.next(i, s, a)
                # print(s,s_)
                # Update this predator's Q function.
                if(aware and isinstance(s,tuple) and False):
                    if(episode<TIED_PARAMS):
                        Q[s[0]][a.value] += BETA * (r + GAMMA * np.max(Q[s_[0]]) - Q[s[0]][a.value])
                    else:
                        Q[s][a.value] += BETA * (r + GAMMA * np.max(Q[s_]) - Q[s][a.value])
                else:
                    Q[s][a.value] += BETA * (r + GAMMA * np.max(Q[s_]) - Q[s][a.value])
                # QC[s][a.value] +=1
                # Set up state and action for next iteration
                predatorStates[i] = s_
                # Broadcast state to fellow predators.
                if(sharing):
                    env.broadcast(i)

            # Increment the timestep
            timestep += 1

        # Re-initialize the environment at the
        # end of the episode
        # if(aware):
        #     if(episode==TIED_PARAMS):
        #         for i, (Q) in enumerate(qList):
        #             for ps in product(xRange, yRange):
        #                 Q[(i,ps)] = np.copy(Q[i])
        #                 for ps_ in product(xRange, yRange):
        #                     Q[(ps,ps_)] = np.copy(Q[ps])
        #                     # Q[ps] = np.zeros(nActions)
        env.initialize()
        episode += 1
        if(episode%100==0):
            last_t = 0 if len(ts)==0 else ts[-1]
            # print("episode %d\r"%(episode),end="")
            if(episode<=1000):
                print("%d episode - %d\tepisode length = %d\taverage steps/eps = %d\taverage last 50 eps = %d"%(int(sharing),episode,timestep-last_t,timestep//episode,(timestep-ts[-50])//50))
            elif(episode<=5000):
                print("%d episode - %d\tepisode length = %d\taverage steps/eps = %d\taverage last 1000 eps = %d"%(int(sharing),episode,timestep-last_t,timestep//episode,(timestep-ts[-1000])//1000))
            else:
                print("%d episode - %d\tepisode length = %d\taverage steps/eps = %d\taverage last 5000 eps = %d"%(int(sharing),episode,timestep-last_t,timestep//episode,(timestep-ts[-5000])//5000))
        es.append(episode)
        ts.append(timestep)
        avg.append(timestep*1.0/episode)
    print("End")
    return qList, es, ts, avg

def Test_run_joint(env, stop, qList, seed,aware=False,sharing=False):
    gen = np.random.RandomState(seed)
    nActions = len(env.actions)
    timestep = 0
    episode = 0

    ts = []
    es = []
    avg=[]
    while stop(episode):
        # Initialization of predator states
        predatorStates = [i for i in range(env.nPredator)]

        while not env.jointTerminate():

            aList = []
            # Selecting action for each predator
            for i in range(env.nPredator):
                Q = qList[i]
                s = predatorStates[i]
                a = env.selectAction(Q, s, T/100.0)
                aList.append(a)

            # Move the prey.
            env.movePrey()
            env.movePredator(aList)
            for i in range(env.nPredator) :
                Q = qList[i]
                s = predatorStates[i]
                a = aList[i]
                # Get next state and reward
                if(aware):
                    s_, r = env.joint_next(i, s, a)
                else:
                    s_, r = env.next(i, s, a)
                # print(s,s_)
                # Update this predator's Q function.
                # Q[s][a.value] += BETA * (r + GAMMA * np.max(Q[s_]) - Q[s][a.value])
                # QC[s][a.value] +=1
                # Set up state and action for next iteration
                predatorStates[i] = s_
                # Broadcast state to fellow predators.
                if(sharing):
                    env.broadcast(i)

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
    return es, ts, avg
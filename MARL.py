import numpy as np
from GridWorld import GridWorld, Action
import matplotlib.pyplot as plt
from itertools import product
from Utils import *
from QL import *
import sys
import pprint

def main () :
    nPrey=1
    nPredator=2
    rows=10
    cols=10
    perceptWindow=4
    reward_scheme=1
    env = GridWorld(nPrey=nPrey,nPredator=nPredator,rows=rows,cols=cols,perceptWindow=perceptWindow,seed=10,reward_scheme=reward_scheme)
    if(sys.argv[1]=="Train"):
        qList_I, es1, ts1, avg1 = QLearning(env, lambda x : x < 10000, 0,sharing=False)
        qList_S, es2, ts2, avg2 = QLearning(env, lambda x : x < 10000, 0,sharing=True)
        save_obj(qList_I,"Independent%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        save_obj(qList_S,"Shared%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
    elif(sys.argv[1]=="NoTrain"):
        qList_I=load_obj("Independent%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        qList_S=load_obj("Shared%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
    esI, tsI, avgI = Test_run(env, lambda x : x < 10000,qList_I, 22,sharing=True)
    esS, tsS, avgS = Test_run(env, lambda x : x < 10000,qList_S, 22,sharing=True)
    iQL  = plt.scatter(esI, avgI, c='red',s=10)
    ssQL = plt.scatter(esS, avgS, c='blue',s=10)
    iQL.set_label("Independent")
    ssQL.set_label("2 Predators, 1 Prey, Comm as Act")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Avergae TimeSteps")
    plt.legend()
    plt.show()
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(qList_S)
    # for i in range(10):
    env.simulateTrajectory(qList_S,sharing=True)

if __name__ == "__main__" : 
    main()

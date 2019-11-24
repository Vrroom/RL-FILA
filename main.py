import numpy as np
from GridWorld import GridWorld, Action
import matplotlib.pyplot as plt
from itertools import product
from Utils import *
from QL import *
from QL_joint import *
import sys
import pprint
import click

@click.command()
@click.argument('Termination', type=click.Choice(['single', 'joint']))
@click.option('--train', type=bool, default=True,help='Train the agents')
@click.option('--train_graph', type=bool, default=True,help='Plot the Training graph')
@click.option('--test', type=bool, default=True,help='Test the Agents')
@click.option('--test_graph', type=bool, default=True,help='Plot the Testing graph')
@click.option('--smart_prey', type=bool, default=True,help='Smart prey or random walking prey')
@click.option('--no_predators', type=int, default=2,help='No. of Predators in the grid')
@click.option('--no_prey', type=int, default=2,help='No. of Predators in the grid')
@click.option('--grid_size', type=int, default=10,help='Size of the grid')
@click.option('--percept_pred', type=int, default=4,help='Perceptual Window of Predator')
@click.option('--percept_prey', type=int, default=4,help='Perceptual Window of Prey')
@click.option('--save_weights', type=bool, default=True,help='Save Weights in directory Obj')
@click.option('--load_weights', type=str, default="",help='Load the saved weigths')

def run(train,train_graph,test,test_graph,smart_prey,no_predators,no_prey,grid_size,percept_pred,percept_prey,save_weights,load_weights) :
    nPrey=no_prey
    nPredator=no_predators
    rows=grid_size
    cols=grid_size
    perceptWindow=percept_pred
    reward_scheme=1
    if(smart_prey):
        sm="2"
    else:
        sm="0"
    env = GridWorld(nPrey=nPrey,nPredator=nPredator,rows=rows,cols=cols,perceptWindow=perceptWindow,seed=10,reward_scheme=reward_scheme,prey_window=percept_prey,smart_prey=sm)
    if(train):
        qList_I, esI, tsI, avgI = QLearning(env, lambda x : x < 20000, 0,sharing=False)
        qList_S, esS, tsS, avgS = QLearning(env, lambda x : x < 20000, 0,sharing=True)
        if(save_weights)
        save_obj(qList_I,"Independent%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        print("Saved weigths ")
        save_obj(qList_S,"Super_Smart_Shared%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
    elif(sys.argv[1]=="NoTrain"):
        qList_I=load_obj("Super_Smart_Independent%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        qList_S=load_obj("Super_Smart_Shared%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        # esI, tsI, avgI = Test_run(env, lambda x : x < 10000,qList_I, 22,sharing=False)
        esS, tsS, avgS = Test_run(env, lambda x : x < 10000,qList_S, 22,sharing=True)
    print(avgI[-1],avgS[-1])
    # print(qList_S)
    # exit()
    # iQL  = plt.scatter(esI, avgI, c='red',s=10)
    # ssQL = plt.scatter(esS, avgS, c='blue',s=10)
    # iQL.set_label("Independent")
    # ssQL.set_label("2 Predators, 2 Prey, Share State")
    # plt.xlabel("Episodes")
    # plt.ylabel("Cumulative Average TimeSteps")
    # plt.legend()
    # plt.show()
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(qList_S)
    for i in range(10):
        env.simulateTrajectory(qList_S,sharing=True)

def run_joint(train,train_graph,test,test_graph,smart_prey,no_predators,no_prey,grid_size,percept_pred,percept_prey,save_weights,load_weights) :
    nPrey=1
    nPredator=2
    rows=10
    cols=10
    perceptWindow=4
    reward_scheme=2
    env = GridWorld(nPrey=nPrey,nPredator=nPredator,rows=rows,cols=cols,perceptWindow=perceptWindow,seed=10,reward_scheme=reward_scheme,smart_prey="2")

    if(sys.argv[1]=="Train"):
        # qList_I=load_obj("Independent_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        # qList_S=load_obj("Share_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        # qList_A=load_obj("Aware_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        # qList_A_S=load_obj("Share_Aware_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        plt.title("Training %d"%(perceptWindow))
        # qList_I, esI, tsI, avgI = QLearning_joint(env, lambda x : x < 25000, 0,aware=False,sharing=False,pretrain=True,_qList=qList_I)
        # qList_S, esS, tsS, avgS = QLearning_joint(env, lambda x : x < 25000, 0,aware=False,sharing=True,pretrain=True,_qList=qList_S)
        # qList_A, esA, tsA, avgA = QLearning_joint(env, lambda x : x < 25000, 0,aware=True,sharing=False)
        qList_A_S, esA_S, tsA_S, avgA_S = QLearning_joint(env, lambda x : x < 25000, 0,aware=True,sharing=True)
        save_obj(qList_I,"Super_Smart_Independent_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        save_obj(qList_S,"Super_Smart_Share_Aware_joint_re_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        save_obj(qList_A,"Super_Smart_Aware_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        save_obj(qList_A_S,"Super_Smart_Share_Aware_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
    elif(sys.argv[1]=="NoTrain"):
        plt.title("Testing %d"%(perceptWindow))
        qList_I=load_obj("Super_Smart_Independent_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        qList_S=load_obj("Super_Smart_Share_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        qList_A=load_obj("Super_Smart_Aware_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        qList_A_S=load_obj("Super_Smart_Share_Aware_joint_%d_%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow,reward_scheme))
        print("Qvalues loaded")
        esI, tsI, avgI = Test_run_joint(env, lambda x : x < 10000,qList_I, 34,aware=False,sharing=False)
        esS, tsS, avgS = Test_run_joint(env, lambda x : x < 10000,qList_S, 34,aware=False,sharing=True)
        esA, tsA, avgA = Test_run_joint(env, lambda x : x < 10000,qList_A, 34,aware=True,sharing=False)
        esA_S, tsA_S, avgA_S = Test_run_joint(env, lambda x : x < 10000,qList_A_S, 34,aware=True,sharing=True)
        print(  avgA_S[-1])
    iQL  = plt.scatter(esI, avgI, c='red',s=10)
    sQL  = plt.scatter(esS, avgS, c='blue',s=10)
    aQL = plt.scatter(esA, avgA, c='green',s=10)
    a_sQL = plt.scatter(esA_S, avgA_S, c='orange',s=10)
    iQL.set_label("Independent")
    sQL.set_label("Shared")
    aQL.set_label("Aware")
    a_sQL.set_label("Shared and Aware")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Average TimeSteps")
    plt.legend()
    plt.show()
    # # pp = pprint.PrettyPrinter(indent=4)
    # # pp.pprint(qList_S)
    for i in range(10):
    #     env.simulateTrajectory(qList_I,aware=False,sharing=False,termination="1")
        # env.simulateTrajectory(qList_S,aware=False,sharing=True,termination="1")
        env.simulateTrajectory(qList_A,aware=True,sharing=False,termination="1")
        # env.simulateTrajectory(qList_A_S,aware=True,sharing=True,termination="1")


def main(Termination,train,train_graph,test,test_graph,smart_prey,no_predators,no_prey,grid_size,percept_pred,percept_prey,save_weights,load_weights):
    if(Termination=="single"):
        run(train,train_graph,test,test_graph,smart_prey,no_predators,no_prey,grid_size,percept_pred,percept_prey,save_weights,load_weights)
    else:
        run_joint(train,train_graph,test,test_graph,smart_prey,no_predators,no_prey,grid_size,percept_pred,percept_prey,save_weights,load_weights)

if __name__ == "__main__" :
    main()
    # print("Hello")
    # print(Termination)

import numpy as np
from GridWorld import GridWorld, Action
import matplotlib.pyplot as plt
from itertools import product
from Utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class learning_nn(nn.Module):
    def __init__(self,state_var_size,num_actions,l,learning_rate=3e-3):
        super(learning_nn, self).__init__()

        self.output_dim = num_actions
        self.input_dim = state_var_size
        self.l = l

        self.l1 = nn.Linear(self.input_dim,self.l,bias=True)
        self.out_layer = nn.Linear(self.l,self.output_dim,bias=True)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self,inp):
        f1 = F.relu(self.l1(inp))
        actions = F.softmax(self.out_layer(f1),dim=1)
        return actions

    def get_action(self,inp):
        state = torch.from_numpy(inp).float().unsqueeze(0)
        probs = self.forward(torch.tensor(state,requires_grad=True))
        highest_prob_action = np.random.choice(self.output_dim, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


# EPLEN=200
def gtmat(le):
    _GtVec = GAMMA**np.arange(le)
    _UPTMAT = np.zeros((le,le))
    _UPTMAT[np.triu_indices(le)] = 1.0
    _GtMat = np.stack([np.roll(_GtVec,i) for i in range(le)])*_UPTMAT
    return _GtMat
def Policy_Grad(env, stop, seed, sharing=False,EPLEN=200):
    _GtMat = gtmat(EPLEN)
    gen = np.random.RandomState(seed)
    nActions = len(env.actions)
    # Each predator has own action.
    policy_networks = [learning_nn(2,5,5) for _ in range(env.nPredator)]

    timestep = 0
    episode = 0

    ts = []
    es = []
    avg=[]
    while stop(episode):
        predatorStates = [i for i in range(env.nPredator)]
        log_probs=[[] for _ in range(env.nPredator)]
        rewards = [[] for _ in range(env.nPredator)]
        while not env.terminate():

            aList = []
            for i in range(env.nPredator):
                s = predatorStates[i]
                if(isinstance(s, tuple)):
                    net_inp = np.asarray(s).astype(np.float32)
                else:
                    net_inp = gen.rand(2).astype(np.float32)

                a,log_prob = policy_networks[i].get_action(net_inp)
                aList.append(a)
                log_probs[i].append(log_prob)

            # Move the prey.
            env.movePrey()

            for i in range(env.nPredator) :
                s = predatorStates[i]
                a = aList[i]
                s_, r = env.next(i, s, a)
                rewards[i].append(r)
                predatorStates[i] = s_
                if(sharing):
                    env.broadcast(i)
            timestep += 1

        ####
        #Policy Update
        ####
        rewards = np.asarray(rewards).astype(np.float32) ## nPredator x ts
        ep_len = rewards.shape[1]
        if(ep_len<=EPLEN):
            GtMat = _GtMat[:ep_len,:ep_len]
        else:
            _GtMat = gtmat(ep_len)
            EPLEN = ep_len
            GtMat = _GtMat # eplen x eplen
        Gt = np.sum(GtMat[np.newaxis,:,:]*rewards[:,:,np.newaxis],axis=2) # nPredator x ts
        norm_GT = (Gt - np.mean(Gt,axis=1)[:,np.newaxis])/(np.std(Gt,axis=1)+1e-9)[:,np.newaxis]
        norm_GT = torch.FloatTensor(norm_GT)
        for i in range(env.nPredator):
            policy_networks[i].optimizer.zero_grad()
            pGrad = torch.stack([-1*lp*Gt for lp,Gt in zip(log_probs[i],norm_GT[i,:])]).sum()
            pGrad.backward()
            policy_networks[i].optimizer.step()
        env.initialize()

        episode += 1
        if(episode%100==0):
            last_t = 0 if len(ts)==0 else ts[-1]
            print("%d episode - %d\tepisode length = %d\taverage steps/eps = %d\taverage last 50 eps = %d"%(int(sharing),episode,timestep-last_t,timestep//episode,(timestep-ts[-50])//50))
            print(list(policy_networks[0].parameters())[0])
        es.append(episode)
        ts.append(timestep)
        avg.append(timestep//episode)
    print("End")
    return policy_networks, es, ts, avg

def main () :
    nPrey=1
    nPredator=2
    rows=10
    cols=10
    perceptWindow=4
    env = GridWorld(nPrey=nPrey,nPredator=nPredator,rows=rows,cols=cols,perceptWindow=perceptWindow,seed=10)
    pn,es,ts,avg = Policy_Grad(env,lambda x : x < 2000,0,sharing=False)
    # if(sys.argv[1]=="Train"):
    #     qList_I, es1, ts1, avg1 = QLearning(env, lambda x : x < 2000, 0,sharing=False)
    #     qList_S, es2, ts2, avg2 = QLearning(env, lambda x : x < 100000, 0,sharing=True)
    #     save_obj(qList_I,"Independent%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow))
    #     save_obj(qList_S,"Shared%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow))
    # elif(sys.argv[1]=="NoTrain"):
    #     qList_I=load_obj("Independent%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow))
    #     qList_S=load_obj("Shared%d_%d_%d_%d"%(nPrey,nPredator,rows,perceptWindow))

    # esI, tsI, avgI = Test_run(env, lambda x : x < 10000,qList_I, 55,sharing=True)
    # esS, tsS, avgS = Test_run(env, lambda x : x < 10000,qList_S, 55,sharing=True)
    # iQL  = plt.scatter(esI, avgI, c='red',s=10)
    # ssQL = plt.scatter(esS, avgS, c='blue',s=10)
    # iQL.set_label("Independent")
    # ssQL.set_label("2 Predators, 1 Prey, Share State")
    # plt.xlabel("Episodes")
    # plt.ylabel("Cumulative Avergae TimeSteps")
    # plt.legend()
    # plt.show()
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(qList_I)
    # for i in range(10):
    # env.simulateTrajectory(qList_I,sharing=True)

if __name__ == "__main__" : 
    main()


from enum import Enum
import numpy as np
import math
from scipy.special import softmax
from Visualize import visualizeTrajectory
from Utils import *

class Action (Enum) :
    Left  = 0
    Right = 1
    Up    = 2
    Down  = 3
    Stay  = 4

# TODO : KILL PREY 


class Predator : 

    def __init__ (self, pId, pos) :
        self.pId = pId
        self.pos = pos
        self.view = pId
        self.state = pId
        self.knowledge = dict()

    def setState(self, s) :
        self.state = s

    def getState(self) :
        return self.state

    def setView(self, s) :
        self.view = s

    def getView(self) :
        return self.view

    def setPosition(self, pos) : 
        self.pos = pos

    def getPosition(self) :
        return self.pos

    def updateKnowledge(self, otherPredator) : 
        # If the other predator has something 
        # meaningful to contribute (in the form of a
        # finite perceptual state), then update this
        # predator's knowledge.
        s_ = otherPredator.getView()
        # print(s_)
        if isinstance(s_, tuple) :
            # print("feeling helped")
            self.knowledge[otherPredator.pId] = s_
        else :
            self.knowledge[otherPredator.pId] = None

def manhattanDistance(p1, p2) : 
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class GridWorld :
    
    def __init__ (self,rows=20,cols=20,nPredator=5,nPrey=2,perceptWindow=2,seed=0,reward_scheme=0) :
    
        self.rows = rows
        self.cols = cols
        self.nPredator = nPredator
        self.nPrey = nPrey
        self.gen = np.random.RandomState(seed)
        self.actions = [Action.Left, Action.Right, Action.Up, Action.Down, Action.Stay]
        self.perceptWindow = perceptWindow
        self.reward_scheme=reward_scheme
        self.initialize()

    def initialize(self) :
        # Spawn the required number of 
        # preys at random locations on the 
        # board.
        self.preys = []
        for i in range(self.nPrey) :
            x = self.gen.randint(self.rows)
            y = self.gen.randint(self.cols)
            self.preys.append((x,y))

        # Spawn the required number of 
        # predators. Make sure that they don't 
        # spawn at the same place as the prey.
        self.predators = []
        for i in range(self.nPredator) :
            x = self.gen.randint(self.rows)
            y = self.gen.randint(self.cols)
            while (x, y) in self.preys :
                x = self.gen.randint(self.rows)
                y = self.gen.randint(self.cols)
            self.predators.append(Predator(i, (x, y)))

    def broadcast (self, pId, info) : 
        # Broadcast the perceptual state of 
        # the predator at pId to the rest of
        # the predators.
        for pId_, predator in enumerate(self.predators) :
            if pId != pId_:
                self.predators[pId_].knowledge[pId] = info
                # print(self.predators[pId_].knowledge)
                # updateKnowledge(predator)

    def move(self, x, y, a, pId) :
        # Move something on the board with
        # regard to the boundary conditions.
        xNew, yNew = x, y

        if a == Action.Left: 
            yNew = max(0, y - 1) 
        elif a == Action.Right :
            yNew = min(self.cols - 1, y + 1)
        elif a == Action.Up :
            xNew = min(self.rows - 1, x + 1)
        elif a == Action.Down :
            xNew = max(0, x - 1)
        elif a == Action.Stay:
            xNew, yNew = x, y
            if pId != -1:
                s_ = self.predators[pId].getView()
                if isinstance(s_, tuple):
                    self.broadcast(pId, s_)
                else:
                    self.broadcast(pId, None)
            return (xNew, yNew)

        if pId != -1:
            self.broadcast(pId, None)
        return (xNew, yNew)

    def movePrey(self) :
        # To be called at each time step to
        # move the preys by taking a random
        # action. 
        for i in range(self.nPrey) : 
            a = self.gen.choice(self.actions)
            x, y = self.preys[i] 
            self.preys[i] = self.move(x, y, a, -1)


    def next(self, pId, perceptState, a) :
        x, y = self.predators[pId].getPosition()
        # simple movement of the predator (doesn't let it go outside the boundaries)
        self.predators[pId].setPosition(self.move(x, y, a, pId))
        # print(self.predators[pId].getPosition())
        minDist = math.inf

        # This initialization is guaranteed
        # to be outside the perceptual window.
        delX, delY = self.rows + 1, self.cols + 1

        # Check whether the perceptual window
        # contains a prey.
        for p in self.preys :
            p_ = self.predators[pId].getPosition()
            d = manhattanDistance(p, p_)
            if d < minDist :
                minDist = d
                delX = p[0] - p_[0]
                delY = p[1] - p_[1]

        # If a prey is within the perceptual
        # window, then return that as the new state.
        # Otherwise the predator searches in its
        # knowledge source for information collected
        # from some other predator.
        if abs(delX) <= self.perceptWindow and abs(delY) <= self.perceptWindow :
            if delX == 0 and delY == 0 : 
                # Reward of 1 for finding the prey.
                self.predators[pId].setState((delX,delY))
                self.predators[pId].setView((delX,delY))
                return (delX, delY), 1
            else :
                # Otherwise a penalty for wasting step.
                self.predators[pId].setState((delX,delY))
                self.predators[pId].setView((delX,delY))
                return (delX, delY), self.reward((delX,delY))
        else :
            self.predators[pId].setView(pId)
            # Again do the same search.
            minDist = math.inf
            delX, delY = self.rows + 1, self.cols + 1
            for pId_, s_ in self.predators[pId].knowledge.items() :
                # If this predator has a prey 
                # in perceptual window.
                if s_ :
                    curPos = self.predators[pId].getPosition()
                    otherPos = self.predators[pId_].getPosition()
                    preyX = otherPos[0] + s_[0]
                    preyY = otherPos[1] + s_[1]
                    delNew = (preyX - curPos[0], preyY - curPos[1])
                    # print(pId,curPos,otherPos,s_)
                    d = manhattanDistance(delNew, (0, 0))
                    if d < minDist :
                        minDist = d
                        delX, delY = delNew

            if minDist < math.inf :
                self.predators[pId].setState((delX,delY))
                return (delX, delY), self.reward((delX,delY))
            else :
                self.predators[pId].setState(pId)
                return pId, -0.1

    def terminate(self) :
        # Terminate when 1 prey is caught by
        # any predator.
        preySet = set(self.preys)
        predatorSet = set([p.getPosition() for p in self.predators])
        return not preySet.isdisjoint(predatorSet)

    def jointTerminate(self) :
        # Terminate when there are two
        # predators which are very close 
        # (with 1 manhattan distance) to the prey.
        for prey in self.preys :
            closeCount = 0
            for predator in self.predators :
                if manhattanDistance(prey, predator.getPosition()) <= 1 :
                    closeCount += 1
            if closeCount >= 2 :
                return True
        return False

    def reward(self,s):
        if(self.reward_scheme==0):
            return -0.1
        elif(self.reward_scheme==1):
            if(isinstance(s,tuple)):
                return -0.1*((1.0*(abs(s[0]))/self.rows)**2+(1.0*(abs(s[1]))/self.cols)**2)
            else:
                return -0.1
        return -0.1

    def toFrame (self) :
        # Convert the current state of
        # the grid world into a frame for animation.
        frame = np.ones((self.rows, self.cols, 4))
        
        red = np.array([1, 0, 0, 1])
        green = np.array([0, 1, 0, 1])
        blue = np.array([0, 0, 1, 1])

        # +1 is for prey and -1 is for 
        # predator. Hence if a cell has +2, 
        # that means that there are two preys
        # in that cell.
        # for p in self.predators :
        #     x, y = p.getPosition()
        #     rngx = (np.min(0,x-self.perceptWindow),np.max(x+self.perceptWindow,self.rows))
        #     rngy = (np.min(0,y-self.perceptWindow),np.max(y+self.perceptWindow,self.cols))
        for x, y in self.preys :
            frame[x, y] = green
            
        for p in self.predators :
            x, y = p.getPosition()
            if (frame[x, y] == green).all() :
                frame[x, y] = red
            else :
                frame[x, y] = blue

        return frame

    def simulateTrajectory (self, qList, sharing=False) : 
        self.initialize()
        frames = []
        preyCaught = False

        predatorStates = [i for i in range(self.nPredator)]

        while not preyCaught:
            
            frames.append(self.toFrame())
            aList = []

            for i in range(self.nPredator) :
                if(sharing):
                    Q,_ = qList[i]
                else:
                    Q = qList[i]
                s = predatorStates[i]
                a = self.selectAction(Q, s, T)
                aList.append(a)

            self.movePrey()

            for i in range(self.nPredator) :
                if(sharing):
                    Q,_ = qList[i]
                else:
                    Q = qList[i]
                s = predatorStates[i]
                a = aList[i]
                # Get next state and reward
                s_, r = self.next(i, s, a) 
                # Check whether prey has been caught.
                # if(sharing):
                #     self.broadcast(i)
                if s_ == (0, 0) :
                    preyCaught = True

                # Update this predator's Q function.
                # Q[s][a.value] += BETA * (r + GAMMA * np.max(Q[s_]) - Q[s][a.value])
                # Set up state and action for next iteration
                predatorStates[i] = s_

        frames.append(self.toFrame())
        visualizeTrajectory(frames)

    def selectAction (self, Q, s, T=1) :
        # Select action using softmax distribution
        # of Q[s] with temperature T.
        aList = self.gen.choice(self.actions, 1, p=softmax(Q[s] / T))
        return aList[0]

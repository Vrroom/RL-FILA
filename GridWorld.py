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

def manhattanDistance(p1, p2) : 
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class GridWorld :
    
    def __init__ (
        self, 
        rows=10, 
        cols=10, 
        nPredator=4, 
        nPrey=2, 
        perceptWindow=2, 
        seed=0
        ) :
        
        self.rows = rows
        self.cols = cols
        self.nPredator = nPredator
        self.nPrey = nPrey
        self.gen = np.random.RandomState(seed)
        self.actions = [Action.Left, Action.Right, Action.Up, Action.Down]
        self.perceptWindow = perceptWindow
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
            self.predators.append((x, y))


    def move(self, x, y, a) :
        # Move something on the board with
        # regard to the boundary conditions.
        xNew, yNew = x, y

        if a == Action.Left: 
            yNew = max(0, y - 1) 
        elif a == Action.Right :
            yNew = (y + 1) % self.cols
        elif a == Action.Up :
            xNew = (x + 1) % self.rows
        else :
            xNew = max(0, x - 1)

        return (xNew, yNew)

    def movePrey(self) :
        # To be called at each time step to
        # move the preys by taking a random
        # action. 
        for i in range(self.nPrey) : 
            a = self.gen.choice(self.actions)
            x, y = self.preys[i] 
            self.preys[i] = self.move(x, y, a)


    def next(self, pId, perceptState, a) :
        x, y = self.predators[pId]
        self.predators[pId] = self.move(x, y, a)

        minDist = math.inf

        # This initialization is guaranteed
        # to be outside the perceptual window.
        delX, delY = self.rows + 1, self.cols + 1

        for p in self.preys :

            d = manhattanDistance(p, self.predators[pId])
            if d < minDist :
                minDist = d
                delX = p[0] - self.predators[pId][0]
                delY = p[1] - self.predators[pId][1]

        # If a prey is within the perceptual
        # window, then return that as the new state.
        # Otherwise return the unique pId.
        if abs(delX) <= self.perceptWindow and abs(delY) <= self.perceptWindow :
            if delX == 0 and delY == 0 : 
                # Reward of 1 for finding the 
                # prey.
                return (delX, delY), 1
            else :
                # Otherwise a penalty for
                # wasting a step.
                return (delX, delY), -0.1
        else :
            return pId, -0.1

    def toFrame (self) :
        # Convert the current state of
        # the grid world into a frame for animation.
        frame = np.zeros((self.rows, self.cols))
        
        # +1 is for prey and -1 is for 
        # predator. Hence if a cell has +2, 
        # that means that there are two preys
        # in that cell.
        for x, y in self.preys :
            frame[x, y] += 1

        for x, y in self.predators :
            frame[x, y] -= 1 

        return frame

    def simulateTrajectory (self, qList) : 
        self.initialize()
        frames = []
        preyCaught = False

        predatorStates = [i for i in range(self.nPredator)]

        while not preyCaught:
            
            frames.append(self.toFrame())

            for i in range(self.nPredator) :
                Q = qList[i]
                s = predatorStates[i]
                a = self.selectAction(Q, s, T)
                # Get next state and reward
                s_, r = self.next(i, s, a) 
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
            self.movePrey()

        visualizeTrajectory(frames)




    def selectAction (self, Q, s, T=1) :
        # Select action using softmax distribution
        # of Q[s] with temperature T.
        aList = self.gen.choice(self.actions, 1, p=softmax(Q[s] / T))
        return aList[0]





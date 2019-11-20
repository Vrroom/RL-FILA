import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation

def visualizeTrajectory (frames) :	

	fig = plt.figure()
	shape = frames[0].shape
	plot = plt.imshow(np.zeros(shape))

	def init():
	    plot.set_data(frames[0])
	    return [plot]

	def update(j):
		if j < len(frames) :
			plot.set_data(frames[j])
		else :
			plot.set_data(frames[-1])
	    
		return [plot]

	anim = FuncAnimation(fig, update, init_func=init, interval=40, blit=True)
	plt.show()
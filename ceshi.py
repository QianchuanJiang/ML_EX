import numpy as np
from Network_main import neuralNetwork

i = 3
h = 3
o = 3
learning_rate = 0.1

n = neuralNetwork(inputnodes=i, hiddennodes=h, outputnodes=o, learningrate=0.3)
print(n.query([1.0, 0.5, -1.5]))

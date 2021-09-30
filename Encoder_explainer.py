"""
Author: Steve Paul 
Date: 9/22/21 """


import torch as th
import numpy as np
from utils import load_model, move_to

# Create small graph

#Permute it and create a larger graph
size = 20
n_node_props = 2
upper = .3
data_input = {
                "loc" :th.FloatTensor(1, size, n_node_props).uniform_(0, upper),
                "demand":th.FloatTensor(1, size).uniform_(0, upper),
                "depot":th.FloatTensor(1, 1,2).uniform_(0, upper)
              }

permutation = np.random.permutation(size)
loc_perm = th.zeros((1,size, n_node_props))
deadline_perm = th.zeros((1,size))
j=0
for i in permutation:
    loc_perm[0,j,:] = data_input["loc"][0,i,:]
    deadline_perm[0,j] = data_input["demand"][0,i]
    j=j+1

data_input_perm = {
                "loc" :loc_perm,
                "demand":deadline_perm,
                "depot":data_input["depot"]
              }


loc_perm_displacement = th.zeros((1,size, n_node_props))
deadline_perm_displacement = th.zeros((1,size))
j=0
for i in permutation:

    loc_perm_displacement[0,j,:] = data_input["loc"][0,i,:] + upper
    deadline_perm_displacement[0,j] = data_input["deadline"][0,i] + upper
    j=j+1

model, _ = load_model('outputs/mrta_200_test_ready')
model.eval()
embeddings_1, _ = model.embedder(data_input)
embeddings_2, _ = model.embedder(data_input_perm)

ft= 0



#Compute node embedding
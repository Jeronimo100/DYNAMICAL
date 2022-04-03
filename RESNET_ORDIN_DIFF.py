import jax.numpy as jnp
import pandas
from pandas import read_csv
import numpy as np

def mlp(params, inputs):
  # A multi-layer perceptron, i.e. a fully-connected neural network.
  for w, b in params:
    outputs = jnp.dot(inputs, w) + b  # Linear transform
    inputs = jnp.tanh(outputs)        # Nonlinearity
  return outputs

def resnet(params, inputs, depth):
  for i in range(depth):
    outputs = mlp(params, inputs) + inputs
  return outputs

import numpy.random as npr
from jax import jit, grad

resnet_depth = 3
def resnet_squared_loss(params, inputs, targets):
  preds = resnet(params, inputs, resnet_depth)
  return jnp.mean(jnp.sum((preds - targets)**2, axis=1))

def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

# A simple gradient-descent optimizer.
#@jit
def resnet_update(params, inputs, targets):
  grads = grad(resnet_squared_loss)(params, inputs, targets)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]


############################################################
t_nsamples=399+2 # 150
t_space = np.linspace(0,21,t_nsamples)
t_space = t_space[:, None]

inputs = np.reshape(t_space, (401,1)) #dataset
    # jnp.reshape(jnp.linspace(dataset), (10,1))#(-2.0, 2.0, 10), (10, 1))


dataframe = read_csv('N_csv', header=None, engine='python')  #
dataframe = dataframe.drop(dataframe.columns[[0]], axis=1) #df.columns is zero-based pd.Index
dataset = dataframe.values
#dataset = dataset[?]
dataset = dataset.astype('float64')
dataset.shape

targets = dataset # inputs**3 + 0.1 * inputs
#targets = targets[:,0]
############################################################


# Hyperparameters.
layer_sizes = [1, 20, 1] # [1, 20, 1]
param_scale = 1.0
step_size = 0.01
train_iters = 100000 #1000

# Initialize and train.
resnet_params = init_random_params(param_scale, layer_sizes)
for i in range(train_iters):
  resnet_params = resnet_update(resnet_params, inputs, targets)



# Plot results.
import matplotlib.pyplot as plt
#fig = plt.figure(figsize=(6, 4), dpi=150)
plt.figure()
#ax = fig.gca()
#ax.scatter(inputs, targets, lw=0.5, color='green')
plt.plot(inputs, targets, linewidth=1, label='N (Infection) data samples')

fine_inputs = jnp.reshape(jnp.linspace(0, 21.0, 401), (401, 1))
#ax.plot(fine_inputs, resnet(resnet_params, fine_inputs, resnet_depth), lw=0.5, color='blue')
plt.plot(fine_inputs, resnet(resnet_params, fine_inputs, resnet_depth), linewidth=1, label='Neural Netw computed trajectory')

#ax.set_xlabel('t')
#ax.set_ylabel('Trajectory')


#plt.plot(t_space, x6_num_sol, linewidth=1, label='Neural Netw sol x6')
#plt.title('Neural ODEs to fit params')
plt.xlabel('t')

#ax.legend()
plt.legend()

#ax.show()
plt.show()





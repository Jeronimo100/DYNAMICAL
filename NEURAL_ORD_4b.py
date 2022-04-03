import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.optimizers as tfko
from tfdiffeq import odeint  # from torchdiffeq import odeint_adjoint as odeint
from My_Diff_Eq_4b import * 

import pandas
from pandas import read_csv

#pandas.DataFrame(aggr).to_csv("all_data_multi_d.csv", header=None)
dataframe = read_csv('all_data_multi_d_WS_0_4.csv', header=None, engine='python')  # read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataframe = dataframe.drop(dataframe.columns[[0]], axis=1) #df.columns is zero-based pd.Index
dataset = dataframe.values
#dataset = dataset[?]
dataset = dataset.astype('float64')
dataset.shape



t_begin=0.
t_end=1.5
t_nsamples=399 # 150
t_space = np.linspace(0,21,t_nsamples) #(t_begin, t_end, t_nsamples)



#dataset_outs = [tf.expand_dims(an_sol_x(t_space), axis=1), \
#                tf.expand_dims(an_sol_y(t_space), axis=1)]

t_space_tensor = tf.constant(t_space)

###########################################################
"""t_space_tensor = tf.constant(t_space)
x_init = tf.constant([0.], dtype=t_space_tensor.dtype)
y_init = tf.constant([0.], dtype=t_space_tensor.dtype)
u_init = tf.convert_to_tensor([x_init, y_init], dtype=t_space_tensor.dtype) """
###########################################################


#[10,0,0,0,0,1.05]
x1_init = tf.constant([10.], dtype=t_space_tensor.dtype)
x2_init = tf.constant([0.], dtype=t_space_tensor.dtype)
x3_init = tf.constant([0.], dtype=t_space_tensor.dtype)
x4_init = tf.constant([0.], dtype=t_space_tensor.dtype)
x5_init = tf.constant([0.], dtype=t_space_tensor.dtype)
x6_init = tf.constant([1.05], dtype=t_space_tensor.dtype)
u_init = tf.convert_to_tensor([x1_init, x2_init, x3_init, x4_init, x5_init, x6_init], dtype=t_space_tensor.dtype)
print(u_init.shape)
print(t_space_tensor.shape)
print(t_space_tensor[0])
print("\n\n")

#u_init = tf.squeeze(tf.squeeze(tf.convert_to_tensor([[x1_init], [x2_init], [x3_init], [x4_init], [x5_init], [x6_init]], dtype=t_space_tensor.dtype), axis=1), axis=1)

args = [tf.Variable(initial_value=1., name='p' + str(i+1), trainable=True,
          dtype=t_space_tensor.dtype) for i in range(0, 7)]

learning_rate = 0.05
epochs = 100
optimizer = tfko.Adam(learning_rate=learning_rate)

def net():
  return odeint(lambda ts, u0: parametric_ode_system(ts, u0, args),
                  u_init, t_space_tensor)
                #lambda ts, u0: parametric_ode_system(ts, u0), u_init, t_space_tensor)

#tf.convert_to_tensor([x_init, y_init], dtype=t_space_tensor.dtype)

def loss_func(num_sol):
  #col_0 = tf.convert_to_tensor([dataset[0]], dtype="float64") 
  #        tf.expand_dims(tf.convert_to_tensor([dataset[:,1]], dtype="float64"),axis=1)
  col_0 = tf.expand_dims(dataset[:,0],axis=1) 
  col_1 = tf.expand_dims(dataset[:,1],axis=1) 
  col_2 = tf.expand_dims(dataset[:,2],axis=1) 
  col_3 = tf.expand_dims(dataset[:,3],axis=1) 
  col_4 = tf.expand_dims(dataset[:,4],axis=1) 
  col_5 = tf.expand_dims(dataset[:,5],axis=1) 
   
  return tf.reduce_sum(tf.square(col_0[:399,:] - num_sol[:, 0])) + \
         tf.reduce_sum(tf.square(col_1[:399,:] - num_sol[:, 1])) + \
         tf.reduce_sum(tf.square(col_2[:399,:] - num_sol[:, 2])) + \
         tf.reduce_sum(tf.square(col_3[:399,:] - num_sol[:, 3])) + \
         tf.reduce_sum(tf.square(col_4[:399,:] - num_sol[:, 4])) + \
         tf.reduce_sum(tf.square(col_5[:399,:] - num_sol[:, 5])) 
         
  """+ \
         tf.reduce_sum(tf.square(dataset_outs[1] - num_sol[:, 1])) + \
                                                                   + \
                                                                   + \
                                                                   + \ """ 
                                                                   
for epoch in range(epochs):
  with tf.GradientTape() as tape:
    num_sol = net()
    loss_value = loss_func(num_sol)
  print("Epoch:", epoch, " loss:", loss_value.numpy())
  grads = tape.gradient(loss_value, args)
  optimizer.apply_gradients(zip(grads, args))


print("Learned parameters:", [args[i].numpy() for i in range(0, 7)])
num_sol = net()
x1_num_sol = num_sol[:, 0].numpy()
x2_num_sol = num_sol[:, 1].numpy()
x3_num_sol = num_sol[:, 2].numpy()
x4_num_sol = num_sol[:, 3].numpy()
x5_num_sol = num_sol[:, 4].numpy()
x6_num_sol = num_sol[:, 5].numpy()

x1_an_sol = dataset[:399,0] # an_sol_x(t_space)
x2_an_sol = dataset[:399,1]
x3_an_sol = dataset[:399,2]
x4_an_sol = dataset[:399,3]
x5_an_sol = dataset[:399,4]
x6_an_sol = dataset[:399,5]  #y_an_sol = an_sol_y(t_space)

plt.figure()
plt.plot(t_space, x1_an_sol,'--', linewidth=2, label='Dataset samples x1')
plt.plot(t_space, x2_an_sol,'--', linewidth=2, label='Dataset samples x2')
plt.plot(t_space, x3_an_sol,'--', linewidth=2, label='Dataset samples x3')
plt.plot(t_space, x4_an_sol,'--', linewidth=2, label='Dataset samples x4')
plt.plot(t_space, x5_an_sol,'--', linewidth=2, label='Dataset samples x5')
plt.plot(t_space, x6_an_sol,'--', linewidth=2, label='Dataset samples x6')
plt.plot(t_space, x1_num_sol, linewidth=1, label='Neural Netw sol x1')
plt.plot(t_space, x2_num_sol, linewidth=1, label='Neural Netw sol x2')
plt.plot(t_space, x3_num_sol, linewidth=1, label='Neural Netw sol x3')
plt.plot(t_space, x4_num_sol, linewidth=1, label='Neural Netw sol x4')
plt.plot(t_space, x5_num_sol, linewidth=1, label='Neural Netw sol x5')
plt.plot(t_space, x6_num_sol, linewidth=1, label='Neural Netw sol x6')
plt.title('Neural ODEs to fit params')
plt.xlabel('t')
plt.legend()
plt.show()

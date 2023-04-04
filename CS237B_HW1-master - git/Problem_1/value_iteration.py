import os, sys, pdb, math, pickle, time
from turtle import shapetransform
from unicodedata import ucd_3_2_0

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

from utils import generate_problem, visualize_value_function

from collections.abc import Mapping


def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    n = problem["n"]
    m = problem["m"]
    pos2idx = problem["pos2idx"]
    idx2pos = problem["idx2pos"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.zeros([sdim,1])
    assert terminal_mask.ndim == 1 and reward.ndim == 2

    ite = 1000 #1000
    # perform value iteration
    for it in range(ite):
        ######### Your code starts here #########
        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid state

        # Ts is a 4 element python list of transition matrices for 4 actions
        # reward has shape [sdim, 4] - represents the reward for each state
        # action pair
        # terminal_mask has shape [sdim] and has entries 1 for terminal states
        # compute the next value function estimate for the iteration
        # compute err = tf.linalg.norm(V_new - V_prev) as a breaking condition
        err = 0

        # V_new = tf.reduce_max([np.dot(gam*Ts[u],V_numpy.flatten()) + reward[:,0] for u in range(adim)])
        # err = max(err, tf.linalg.norm(V_new - V_numpy))
        # V_numpy = V_new[0].numpy()
        # V_new = [tf.reduce_max(Ts[u]@V + tf.reshape(reward[:,u],[400,1])) for u in range(adim)]

        V_new = [gam*Ts[u]@V + tf.reshape(reward[:,u],[400,1]) for u in range(adim)]
        V_new2 = tf.maximum(V_new[0],V_new[1],V_new[2])
        V_new = tf.maximum(V_new2, V_new[3])

        err = max(err, tf.linalg.norm(V_new - V))
        V = V_new

        ######### Your code ends here ###########
        if err < 1e-7:
            V = tf.reshape(V,[400])
            break

    #find optimal policy part iii
    N = 50 #100
    x0 = pos2idx[0, 0]
    x_policy = [idx2pos[x0][0]]
    y_policy = [idx2pos[x0][1]]

    x_eye, sig = np.array([15, 7]), 1
    w_fn = lambda x: np.exp(-np.linalg.norm(np.array(x) - x_eye) / sig ** 2 / 2)
    xclip = lambda x: min(max(0, x), m - 1)
    yclip = lambda y: min(max(0, y), n - 1)
    for p in range(N):
        z = idx2pos[x0]

        #right up left down
        right = (xclip(z[0] + 1), yclip(z[1] + 0))
        up = (xclip(z[0] + 0), yclip(z[1] + 1))
        left = (xclip(z[0] - 1), yclip(z[1] + 0))
        down = (xclip(z[0] + 0), yclip(z[1] - 1))

        # create list of isndices for loop
        indices = [pos2idx[right[0], right[1]],pos2idx[up[0], up[1]],pos2idx[left[0], left[1]],pos2idx[down[0], down[1]]]

        # find best potential move
        V_4 = [gam*Ts[u][x0][indices[u]]*V[indices[u]] + reward[x0][u] for u in range(adim)]
        index = indices[tf.argmax(V_4).numpy()]

        choice = [w_fn(x0)/3, w_fn(x0)/3, w_fn(x0)/3, w_fn(x0)/3]
        choice[tf.argmax(V_4).numpy()] = 1 - w_fn(x0)

        samples = tf.random.categorical(tf.math.log([choice]), 1)
        print("samples", samples)
        pick = tf.gather(indices, samples).numpy().item()

        x0 = pick
        x_policy.append(idx2pos[x0][0])
        y_policy.append(idx2pos[x0][1])

        if x0 == 389:
            break

    return V, x_policy, y_policy


# value iteration ##############################################################
def main():
    # generate the problem
    problem = generate_problem()
    n = problem["n"]
    sdim, adim = n * n, 1

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[problem["pos2idx"][19, 9]] = 1.0
    terminal_mask = tf.convert_to_tensor(terminal_mask, dtype=tf.float32)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[problem["pos2idx"][19, 9], :] = 1.0
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    gam = 0.95
    V_opt, xp, yp = value_iteration(problem, reward, terminal_mask, gam)

    plt.figure(213)
    visualize_value_function(np.array(V_opt).reshape((n, n)))
    plt.plot(xp,yp,color='#FF0000')
    plt.title("value iteration")
    plt.show()

if __name__ == "__main__":
    main()

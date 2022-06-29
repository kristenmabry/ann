import pandas as pd
import numpy as np
import random
import results

classes = ['a', 'c', 'e', 'm', 'n', 'o', 'r', 's', 'x', 'z']
def elemMatMul(x: np.matrix, y: np.matrix):
    return np.matrix(np.diag(np.outer(x, y))).T

def generateOutput(input: np.matrix, hidden_weights: np.matrix, act_func, outer_weights: np.matrix):
    net_j = np.matmul(hidden_weights, input)
    y_j = act_func(net_j)
    net_k = np.matmul(outer_weights, y_j)
    return net_j, y_j, net_k, act_func(net_k)

def annClassify(input: pd.DataFrame, hidden_weights: np.matrix, act_func, outer_weights: np.matrix):
    input_matrix = np.matrix(input.drop('class', axis=1)).T # convert to 8xn matrix
    net_j = np.matmul(hidden_weights, input_matrix)
    y_j = act_func(net_j)
    net_k = np.matmul(outer_weights, y_j)
    z_k = act_func(net_k)
    guesses = []
    for i in z_k.T:
        index = np.argmax(i)
        guesses.append(classes[index])
    return guesses

def train_network(input: pd.DataFrame, exp_res: np.matrix, act_func, d_act_func, byEpoch: bool = True, learn_rate: float = .1, num_hidden_nodes: int = 4, addBias: bool = False, useMomentum: bool = False):
    n_iter = 0
    max_iter = 2000
    tot_err = 100
    max_err = 16 if byEpoch else 2
    min_change = .0005

    w_ji = np.loadtxt('wji.txt', usecols=range(8))
    w_kj = np.loadtxt('wkj.txt', usecols=range(4))

    # add hidden node(s)
    for i in range(num_hidden_nodes - 4):
        newRow = (np.random.randint(2000, size=8) - 1000)/1000
        w_ji = np.insert(w_ji, 0, [newRow], axis= 0)
        newCol = (np.random.randint(2000, size=10) - 1000)/1000
        w_kj = np.insert(w_kj, 0, [newCol], axis=1)
    # remove hidden nodes
    for i in range(4 - num_hidden_nodes):
        w_ji = np.delete(w_ji, 0, axis=0)
        w_kj = np.delete(w_kj, 0, axis=1)
    # add bias to input
    if addBias:
        biasCol = (np.random.randint(2000, size=4) - 1000)/1000
        w_ji = np.insert(w_ji, 0, [biasCol], axis= 1)

    old_w_kj = []
    old_del_w_kj = np.zeros((10, num_hidden_nodes))
    old_w_ji = []
    old_del_w_ji = np.zeros((num_hidden_nodes, 8 + (1 if addBias else 0)))
    d_w = [1, 1]
    delta_k = 0
    delta_j = 0
    y_j = 0
    z_k = 0
    net_k = 0
    net_j = 0
    alpha = .1 if useMomentum else 0

    jws = []
    perc_error = []

    while ((n_iter < max_iter) and (tot_err > max_err) and (sum(d_w) > min_change)):
        tot_err = 0
        n_iter += 1
        old_w_kj = w_kj.copy()
        old_w_ji = w_ji.copy()

        int_array = list(range(100))
        random.shuffle(int_array)
        update_wkj = np.zeros((10, num_hidden_nodes))
        update_wji = np.zeros((num_hidden_nodes, 8 + (1 if addBias else 0)))
        for i in int_array:
            # get initial results
            x_i = np.matrix(input.drop('class', axis=1).loc[i]).T
            exp_i = int(i/10)
            t_k = np.matrix(exp_res[:, exp_i]).T
            net_j, y_j, net_k, z_k = generateOutput(x_i, w_ji, act_func, w_kj)

            tot_err += np.square(t_k - z_k).sum()
            
            # get delta k and j
            delta_k = elemMatMul((t_k - z_k), d_act_func(net_k))
            sum_k = np.zeros((1, num_hidden_nodes))
            for j in range(10):
                sum_k += w_kj[j, :] * delta_k[j, 0]
            delta_j = elemMatMul(d_act_func(net_j), sum(sum_k))
            
            if byEpoch:
                # update weights
                update_wkj += learn_rate * np.outer(delta_k, y_j)
                update_wji += learn_rate * np.outer(delta_j, x_i)
            else:
                # update weights
                update_wkj = learn_rate * np.outer(delta_k, y_j)
                update_wji = learn_rate * np.outer(delta_j, x_i)
                w_kj += (1-alpha)*update_wkj + alpha*old_del_w_kj
                w_ji += (1-alpha)*update_wji + alpha*old_del_w_ji
                old_del_w_ji = update_wji.copy()
                old_del_w_kj = update_wkj.copy()
                

        if byEpoch:
            # update weights
            w_kj += (1-alpha)*update_wkj + alpha*old_del_w_kj
            w_ji += (1-alpha)*update_wji + alpha*old_del_w_ji
            old_del_w_ji = update_wji.copy()
            old_del_w_kj = update_wkj.copy()
            
        d_w = [np.square(old_w_kj - w_kj).sum(), np.square(old_w_ji - w_ji).sum()]
        jws.append(tot_err)
        perc_error.append(results.getPercentError(input, annClassify(input, w_ji, act_func, w_kj)))
        print(tot_err, perc_error[-1])
    
    print(n_iter, d_w)

    return w_ji, w_kj, jws, perc_error
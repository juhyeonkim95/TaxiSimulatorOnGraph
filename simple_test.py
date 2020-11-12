import matplotlib.pyplot as plt

from math_utils import softmax, softmax_pow
from graph_utils import *

'''
This is a code for supplementary material.
'''


def get_policy(action_values, strategy, k):
    x = None
    if strategy == 'pow':
        x = softmax_pow(action_values, k)
    elif strategy == 'exp':
        x = softmax(action_values, k)
    x[x < 1e-6] = 0
    x[x > 1 - 1e-6] = 1
    return x


def simulate(strategy, k):
    q_values = np.array([1, 1], dtype=np.float32)
    N_calls = np.array([3, 7], dtype=np.float32)
    policy = None
    total_reward = 0
    N = 10
    alpha = 1
    difference = 0
    for i in range(10000):
        policy = get_policy(q_values, strategy, k)
        pi_N = N * policy
        r1 = min(1, N_calls[0] / max(pi_N[0], 1e-8))
        r2 = min(1, N_calls[1] / max(pi_N[1], 1e-8))
        reward = np.array([r1, r2], np.float32)
        total_reward_temp = np.dot(pi_N, reward)

        total_reward = total_reward_temp
        new_q = reward # + 0.9 * q_values
        difference = np.sum(np.abs(q_values - new_q))
        q_values = (1- alpha) * q_values + alpha * new_q
        if i > 9990:
            print("Q", q_values)
            print("Policy", policy)

    if abs(difference) > 0.01:
        print('-------------------------------')
        print(strategy, k)
        print("Policy", policy)
        print("Q", q_values)
        print("Total reward", total_reward)
        print("difference", difference)

    return total_reward


if __name__ == '__main__':
    c = 0.01
    cs = []
    results_exp = []
    results_pow = []
    for i in range(30):
        exp_r = simulate('exp', c)
        pow_r = simulate('pow', c)
        results_exp.append(exp_r)
        results_pow.append(pow_r)
        cs.append(c)
        if c > 1:
            c *= 1.1
        else:
            c *= 2

    plt.plot(cs, results_exp, label='exp')
    plt.plot(cs, results_pow, label='pow')
    plt.xscale('log')
    plt.xlabel(r'$\beta$')
    plt.ylabel('total reward')
    plt.legend()
    plt.show()

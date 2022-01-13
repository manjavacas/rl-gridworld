import gym
import gridworld

import numpy as np


'''GridWorld values prediction by using Monte Carlo'''


def main():

    cells = np.array([['S', 'W', '-', '-', '-', 'G'],
                      ['-', 'W', 'W', '-', 'W', 'W'],
                      ['-', '-', '-', '-', 'W', '-'],
                      ['-', 'W', 'W', '-', '-', '-'],
                      ['-', '-', '-', '-', 'W', '-']])

    env = gym.make('GridWorld-v0', states_matrix=cells)

    values = np.zeros(cells.shape, dtype=float)
    visits = np.zeros(cells.shape, dtype=float)

    N_EPISODES = 10

    # LEARNING VALUES BY MONTECARLO
    for i in range(N_EPISODES):
        env.reset()
        visited = []
        done = False
        R = 0.0
        while not done:
            a = env.action_space.sample()
            obs, reward, done, info = env.step(a)

            if obs not in visited:
                visited.append(obs)

            print(info)
            env.render(mode='human', time_bt_frames=.001)
            R += reward
        print('Total reward for the episode ' + str(i) + ': %.3f' % R)

        # Values update
        for state in visited:
            visits[state[0]][state[1]] += 1
            values[state[0]][state[1]] = round(values[state[0]][state[1]] +
                                               (1/visits[state[0]][state[1]])*(R - values[state[0]][state[1]]), 3)

    print('End! Final values:')
    print(values)


if __name__ == '__main__':
    main()

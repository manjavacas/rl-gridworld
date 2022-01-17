import gym
import gridworld

import numpy as np

N_EPISODES = 100_000
EPSILON = .7
ALPHA = .8
GAMMA = .95


'''Gridworld solved with Q-learning'''


def main():

    grid = np.array([['S', 'W', '-', '-', '-', 'G'],
                     ['-', 'W', 'W', '-', 'W', 'W'],
                     ['-', '-', '-', '-', 'W', '-'],
                     ['-', 'W', 'W', '-', '-', '-'],
                     ['-', '-', '-', '-', 'W', '-']])

    env = gym.make('GridWorld-v0', states_matrix=grid)

    q_values = np.zeros((env.observation_space.n, env.action_space.n))

    # Parse state coordinates to ID (matrix row)
    states_dict = parse_states(grid)

    R = 0.0

    for i in range(N_EPISODES):
        state0, _, _, _ = env.reset()
        done = False

        action0 = choose_action(states_dict[state0], env, q_values)

        while not done:

            state1, reward, done, info = env.step(action0)

            action1 = choose_action(states_dict[state1], env, q_values)

            update_values(states_dict[state0], action0,
                          reward, states_dict[state1], q_values)

            state0 = state1
            action0 = action1

            R += reward

        print('Total reward for the episode ' + str(i) + ': %.3f' % R)

    print('End! Final values:\n' + str(q_values))

    evaluate(env, q_values, states_dict)


def evaluate(env, q_values, states_dict):

    # Change {0:(0,0), 1:(0,1)...} to {(0,0):0, (0,1): 1...}
    states_dict = dict(zip(states_dict.values(), states_dict.keys()))

    best_actions_dict = {}

    for state in range(len(q_values)):
        best_action = np.argmax(q_values[state, :])
        best_actions_dict[states_dict[state]] = best_action

    state, _, _, _ = env.reset()
    done = False
    R = 0.0

    while not done:
        env.render('human', time_bt_frames=.5)
        action = best_actions_dict[state]
        state, reward, done, info = env.step(action)
        R += reward

    print('Total reward for the episode: %.3f' % R)


def choose_action(state, env, q_values, epsilon=EPSILON):
    action = None
    if np.random.uniform(0, 1) > epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_values[state, :])
    return action


def update_values(state0, action0, reward, state1, q_values, alpha=ALPHA, gamma=GAMMA):
    current = q_values[state0, action0]
    target = reward + gamma * max(q_values[state1, :])
    q_values[state0, action0] = current + alpha * (target - current)


def parse_states(grid):
    states_dict = {}
    n = 0
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            states_dict[(x, y)] = n
            n += 1
    return states_dict


if __name__ == '__main__':
    main()

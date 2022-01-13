#!/usr/bin/env python3

''' Deterministic MDP solved with value iteration algorithm '''

import numpy as np
import matplotlib.pyplot as plt

WIDTH = 4
HEIGHT = 4


class State:
    def __init__(self, x, y, id='white', value=0):
        self.x = x
        self.y = y
        self.id = id
        self.value = value

    def get_neighbours(self, states):
        return [s for s in states if abs(self.x - s.x) + abs(self.y - s.y) == 1]


class MDP:
    def __init__(self, states, rewards, gamma=.9):
        self.states = states
        self.rewards = rewards
        self.gamma = gamma

    def value_iteration(self, theta=1e-6):
        i = 0
        while True:
            delta = 0
            for state in self.states:
                if state.id == 'terminal':
                    state.value = 0
                else:
                    v_old = state.value
                    state.value = self.get_value(state)
                    delta = max(delta, abs(v_old - state.value))
            i += 1
            if delta < theta:
                print(f'Converged after {i} iterations!')
                break

    def get_value(self, state):
        neighbours = state.get_neighbours(self.states)
        values = []
        for n in neighbours:
            values.append(1/len(neighbours) *
                          (self.get_reward(n) + self.gamma * n.value))
        return sum(values)

    def get_reward(self, state):
        return self.rewards[state.id]


def visualize(states, title):
    values = np.zeros((HEIGHT, WIDTH))
    for state in states:
        values[state.x][state.y] = state.value

    plt.matshow(values, cmap='YlGnBu_r')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    for state in states:
        plt.text(state.y, state.x, str(round(state.value, 2)),
                 va='center', ha='center')

    plt.show()


if __name__ == '__main__':

    # Problem definition
    states = []
    for x in range(HEIGHT):
        for y in range(WIDTH):
            states.append(State(x, y))

    states[0].id = 'terminal'
    states[-1].id = 'terminal'

    rewards = {'white': -1, 'terminal': 0}

    # Resolution
    mdp = MDP(states, rewards, gamma=1.0)
    mdp.value_iteration()

    # Visualization
    visualize(states, title='Final values')

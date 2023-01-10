import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

from gym import spaces

from time import sleep

ACTIONS = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}


class GridWorldEnv(gym.Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, states_matrix):
        self.states_matrix = states_matrix

        self.height = len(self.states_matrix)
        self.width = len(self.states_matrix[0])

        self.start = self.get_position('S')[0]
        self.goal = self.get_position('G')[0]
        self.walls = self.get_position('W')
        self.blanks = self.get_position('-')

        self.current_position = self.start

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(
            self.height * self.width)

    def step(self, action):

        done = False
        reward = -1

        if action == 0:  # UP
            new_position = (
                self.current_position[0] - 1, self.current_position[1])
        elif action == 1:  # DOWN
            new_position = (
                self.current_position[0] + 1, self.current_position[1])
        elif action == 2:  # LEFT
            new_position = (
                self.current_position[0], self.current_position[1] - 1)
        elif action == 3:  # RIGHT
            new_position = (
                self.current_position[0], self.current_position[1] + 1)
        else:
            print('ERROR: Incorrect action!')

        if new_position in self.blanks or new_position in [self.start, self.goal]:
            self.current_position = new_position

        if self.current_position == self.goal:
            reward = 1
            done = True

        return (self.current_position, reward, done, {'goes_to': self.current_position, 'by_going': ACTIONS[action]})

    def reset(self):
        self.current_position = self.start
        return (self.current_position, 0, False, {})

    def render(self, mode='human', time_bt_frames=.01):
        if mode == 'ansi':
            cellchar = ''
            for x in range(self.height):
                for y in range(self.width):
                    if self.current_position == (x, y):
                        cellchar = 'X'
                    else:
                        cellchar = self.states_matrix[x][y]
                    if y == 0:
                        print('|' + cellchar, end='')
                    elif y == self.width - 1:
                        print(cellchar + '|', end='')
                    else:
                        print('|' + cellchar + '|', end='')
                print()
        elif mode == 'human':
            nrows, ncols = self.height, self.width
            image = np.zeros((nrows, ncols))

            color_codes = {'-': 0, 'S': 1, 'W': 2, 'G': 3, }
            color_map = col.ListedColormap(
                ['white', 'grey', 'black', 'green'])

            for x in range(self.height):
                for y in range(self.width):
                    image[x][y] = color_codes[self.states_matrix[x][y]]

            plt.imshow(image, alpha=.4, cmap=color_map)

            plt.plot(self.current_position[1],
                     self.current_position[0], 'o', markersize=50, color='blue')

            plt.tick_params(left=False,
                            bottom=False,
                            labelleft=False,
                            labelbottom=False)
            plt.ion()
            plt.show()
            plt.pause(time_bt_frames)
            plt.clf()

    def get_position(self, celltype):
        cells = []
        for x in range(self.height):
            for y in range(self.width):
                if self.states_matrix[x][y] == celltype:
                    cells.append((x, y))
        return cells

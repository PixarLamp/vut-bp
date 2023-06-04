# author: Adrián Tulušák
# year: 2020
# file: environment.py
# description: file contains environment for playing simplier version of Scotland Yard,
#              movement methods and states of game

import random
import itertools
from MS_2021 import monte_carlo


class Environment:
    def __init__(self, field_size=5):
        self.mrx = 1
        self.mrx_last_seen = -1
        self.agent1 = 3
        self.agent2 = 7
        self.agents = [3, 7]
        self.epochs = 0
        self.alfabeta_moves = 0

        self.cols = field_size
        self.rows = field_size

        self.reset()

    def reset(self):
        positions = [x for x in range(self.cols * self.rows)]

        self.mrx_last_seen = -1

        self.mrx = random.choice(positions)
        positions.remove(self.mrx)

        self.agent1 = random.choice(positions)
        positions.remove(self.agent1)

        self.agent2 = random.choice(positions)

        self.epochs = 0

        self.alfabeta_moves = 0

    def move_agents(self, alfabeta):
        # every 3th move explore state space with new information about Mr.X
        if self.alfabeta_moves % 3 == 0:
            alfabeta.explore_state_space(self.agent1, self.agent2, self.mrx)

        self.agent1, self.agent2 = alfabeta.choose_new_move_agents()

        self.alfabeta_moves = (self.alfabeta_moves + 1) % 3

    def move_mrx(self, alfabeta):
        new_mrx_position = alfabeta.move_mrx(self.agent1, self.agent2, self.mrx)  # AI vs AI
        # new_mrx_position = player_input.handle_input(self.mrx, self.agent1, self.agent2)        # AI vs player

        if new_mrx_position == -1:
            return False
        self.mrx = new_mrx_position

        if self.alfabeta_moves % 3 == 0:
            self.mrx_last_seen = self.mrx

        return True

    # check if game finished
    def finished(self):
        if (self.agent1 == self.mrx) or (self.agent2 == self.mrx):
            return True
        return False

    def render(self):
        cols = self.cols
        rows = self.rows

        out_map = [[' '] * cols for i in range(rows)]

        if self.mrx_last_seen != -1:
            out_map[self.mrx_last_seen // rows][self.mrx_last_seen % cols] = "-"
        out_map[self.mrx // rows][self.mrx % cols] = "X"
        out_map[self.agent1 // rows][self.agent1 % cols] = "1"
        out_map[self.agent2 // rows][self.agent2 % cols] = "2"

        out = ""
        print("+---------+")
        for row in range(len(out_map)):
            out = "|"

            for column in range(len(out_map[row])):
                if column == len(out_map[row]):
                    out += out_map[row][column] + '|'
                else:
                    out += out_map[row][column] + ':'

            print(out)
        print("+---------+")

        return


# methods for movement #
# UP
def go_up(position, fs=5):
    if (position - fs) < 0:
        return -1
    return position - fs


# DOWN
def go_down(position, fs=5):
    if (position + fs) >= fs * fs:
        return -1
    return position + fs


# LEFT
def go_left(position, fs=5):
    if (position % fs) == 0:
        return -1
    return position - 1


# RIGHT
def go_right(position, fs=5):
    if ((position + 1) % fs) == 0:
        return -1
    return position + 1


# returns list of new positions or -1 if invalid move
def get_valid_moves(position, field_size=5):
    up = go_up(position, fs=field_size)
    down = go_down(position, fs=field_size)
    left = go_left(position, fs=field_size)
    right = go_right(position, fs=field_size)

    return [up, down, left, right]


# returns valid moves according to environment - not according to agents
def get_valid_moves_mrx_vs_player(mrx):
    valid_in_env = get_valid_moves(mrx)
    valid_moves = []

    for move in valid_in_env:
        if move != -1:
            valid_moves.append(move)

    return valid_moves


# returns valid moves according to environment and to agents
def get_valid_moves_mrx(mrx, agents, fs=5):
    valid_in_env = get_valid_moves(mrx, field_size=fs)
    valid_moves = []
    for move in valid_in_env:
        if move != -1 and move not in agents:
            valid_moves.append(move)

    return valid_moves


def get_valid_moves_agents(positions, fs=5):
    agents = []
    for position in positions:
        a_moves = get_valid_moves(position, field_size=fs)
        for a_move in a_moves:
            if a_move in positions[positions.index(position):]:
                a_moves.remove(a_move)
        agents.append(a_moves)

    possible_positions = []

    for agent_moves in itertools.product(*agents):
        if not monte_carlo.has_duplicate(agent_moves) and -1 not in agent_moves:
            possible_positions.append(list(agent_moves))

    return possible_positions

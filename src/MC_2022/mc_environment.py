# author: Michal Sova
# year: 2021
# file: mc_environment.py
# description: Class with environment for the game

# edited: Martin Gerža
# year of edit: 2022
# description of edit: Small changes for better print of game in console and one fix of win_condition.
#                      Changes marked by TODO-MG !!!

import random


class Environment:
    field_size = 5
    mrx_pos = [-1, -1]  # [row, col]
    agents = [[0, 0], [0, 1]]
    mrx_last_seen = 1
    mrx_ls_pos = mrx_pos
    agents_win = False
    last_seen_length = 3
    mrx_visible = False  # TODO-MG added for better determination of Mr. X visibility when printing map in console

    def reset(self, field_size, n_of_agents, last_seen):
        """reset environment with random coordinates of agents and mr X
        :param field_size: size of game field
        :param n_of_agents: number of agents
        :param last_seen: every 'last_seen' move will be mr X visible
        """
        self.mrx_last_seen = 1
        self.mrx_ls_pos = [-1, -1]
        self.field_size = field_size
        self.last_seen_length = last_seen
        self.agents_win = False
        self.mrx_visible = False  # TODO-MG added
        self.agents.clear()
        for agent in range(n_of_agents):
            if agent == 0:
                self.agents.append([random.randint(0, field_size - 1),
                                    random.randint(0, field_size - 1)])  # agents.append([rnd(x), rnd(y)])
            else:
                pos = [random.randint(0, field_size - 1), random.randint(0, field_size - 1)]
                while pos in self.agents:
                    pos = [random.randint(0, field_size - 1), random.randint(0, field_size - 1)]
                self.agents.append(pos)

        self.mrx_pos = [random.randint(0, field_size - 1), random.randint(0, field_size - 1)]
        while self.mrx_pos in self.agents or not self.gen_possible_mrX_moves():  # TODO-MG added scnd part of condition
            self.mrx_pos = [random.randint(0, field_size - 1), random.randint(0, field_size - 1)]

    def set(self, mrx_pos, agents, ls=-1, ls_pos=None):
        """set attributes of environment (recommended after)
        :param mrx_pos: position of mr X
        :param agents: position of agents [[a1x, a1y], [a2x, a2y], ...]
        :param ls: when was mr X last seen
        :param ls_pos: position of mr X, when he was last seen
        """
        if ls_pos is None:
            ls_pos = [-1, -1]
        self.mrx_last_seen = ls
        self.mrx_ls_pos = ls_pos
        self.mrx_pos = mrx_pos
        self.agents.clear()
        for agent in agents:
            self.agents.append(agent)

    def win_condition(self):
        """
        :return True: Agents won, False: MrX won.
        """
        if self.agents_win:
            return True
        for agent in self.agents:
            if self.mrx_pos == agent:
                return True

        # TODO-MG added check of win situation when Mr. X is surrounded by agents --------------------V
        mrX_possible_moves = [move for move in self.gen_possible_mrX_moves() if move not in self.agents]
        if not mrX_possible_moves:
            return True
        # TODO-MG ------------------------------------------------------------------------------------^

        return False

    # person: 0 - mrX, 1 - agent1, 2 - agent2, ...
    def move(self, person, new_position):
        if not (0 <= new_position[0] < self.field_size and 0 <= new_position[1] < self.field_size):
            if person == 0:
                self.agents_win = True
            return False

        if person == 0:
            self.mrx_pos = new_position
            self.mrx_visible = False  # TODO-MG added
            if self.mrx_last_seen % self.last_seen_length == 0:
                self.mrx_ls_pos = self.mrx_pos
                self.mrx_visible = True  # TODO-MG added
        else:
            if new_position in self.agents:
                return False
            self.agents[person - 1] = new_position
        return True

    def print(self):
        print("  | ", end="")
        for num in range(self.field_size):
            print(num, end=" | ")
        print()
        for row in range(self.field_size):
            print(row, '| ', end='')
            for col in range(self.field_size):
                for agent in self.agents:
                    if agent[0] == row and agent[1] == col:
                        if agent == self.mrx_pos:
                            print(str(self.agents.index(agent) + 1) +              # TODO-MG changed (design fix)
                                  ('X' if not self.mrx_visible else 'S'), end='| ')  # TODO-MG changed (design fix)
                        else:
                            print(self.agents.index(agent) + 1, end=' | ')
                        break
                else:
                    if self.mrx_pos[0] == row and self.mrx_pos[1] == col:
                        # TODO-MG: if self.mrx_pos == self.mrx_ls_pos: ...replaced by next row
                        if self.mrx_visible:  # TODO-MG added
                            print('S', end=' | ')
                        else:
                            print('X', end=' | ')
                    elif self.mrx_last_seen > 0 and self.mrx_ls_pos[0] == row and self.mrx_ls_pos[1] == col:
                        print('-', end=' | ')
                    else:
                        print(' ', end=' | ')
            print()

    # TODO-MG added to generate possible moves for Mr. X to check of win situation when  Mr. X surrounded by agents -v
    def gen_possible_mrX_moves(self):
        mrx_pos = self.mrx_pos

        moves = [[mrx_pos[0] + 1, mrx_pos[1]], [mrx_pos[0] - 1, mrx_pos[1]], [mrx_pos[0], mrx_pos[1] + 1],
                 [mrx_pos[0], mrx_pos[1] - 1]]

        return [move for move in moves if (0 <= move[0] < self.field_size) and (0 <= move[1] < self.field_size)]
    # TODO-MG -------------------------------------------------------------------------------------------------------^

# author: Martin Gerža
# year: 2022
# file: MG_monte_carlo.py
# description: Own implementation of MCTS method for both agents and Mr. X

import random as rnd
import math as m
import itertools
import time

# Const for UCB calculation
C = 5.0

# Global variables set from arguments at the start of program
field_size = 0
last_seen_length = 0
max_moves = 0
agents_mcts_type = True


class Node:
    def __init__(self, mrX_pos, agents_pos, act_move, agents_move, visited=0):
        """
        Init function for Node\n
        :param mrX_pos:         Actual possition of Mr. X (Actual known when used for agents)
        :param agents_pos:      Actual possitions of Agents
        :param act_move:        Number of actual move (round)
        :param agents_move:     True when its agents turn, False for Mr. X
        :param visited:         Number of visits for its Node in MCTS iterations
        """
        self.parent = None
        self.leaves = list()
        self.UCB = float('0')
        self.visited = visited
        self.value = 0
        self.wins = 0
        self.agents_pos = [pos for pos in agents_pos]
        self.mrX_pos = mrX_pos
        self.act_move = act_move
        self.agents_move = agents_move

    def select_leaf(self):
        """
        Select func for MCTS using recursion to choose Node for expand and in which Node should simulation be started\n
        :return:                Selected best leaf, that is in this function expanded and will be start for simulation
        """
        if not self.visited:
            return self

        if len(self.leaves) == 0:
            if (self.act_move + 1) < max_moves or (self.act_move < max_moves and not self.agents_move):
                self.expand()
                if len(self.leaves) == 0:
                    return self
                else:
                    return self.leaves[0]

        for leaf in self.leaves:
            if not leaf.visited:
                return leaf

        if not self.leaves:
            return self

        best_leaf = self.leaves[0]
        for leaf in self.leaves:

            leaf.UCB = leaf.wins / leaf.visited + C * m.sqrt(m.log(leaf.parent.visited) / leaf.visited)

            if leaf.UCB > best_leaf.UCB:
                best_leaf = leaf

        return best_leaf.select_leaf()

    def expand(self):
        """
        Expands its own Node, depending on its atributes\n
        :return:                None
        """
        # Part of code for experiment 3
        """
        if self.parent is not None and \
                (self.mrX_pos in self.agents_pos or
                 not [move for move in generate_moves(self.mrX_pos) if move not in self.agents_pos]):
            return
        """
        # End of part of code for exp. 3

        if not self.agents_move:
            mrX_moves = [self.mrX_pos]

            # Part of code for Experiment 3 -- also used in final version
            if agents_mcts_type:
                move = self.act_move + 1
                while move % last_seen_length != 0:
                    old = mrX_moves
                    mrX_moves = []
                    for mov in old:
                        for mo in [move for move in generate_moves(mov) if move not in self.agents_pos]:
                            if mo not in mrX_moves:
                                mrX_moves.append(mo)
                    move -= 1
            # End of part of code for exp. 3

            agents_moves = generate_agent_moves(self.agents_pos)

            # Final version
            if agents_mcts_type:
                agents_moves = get_agents_direction_moves(self.agents_pos, agents_moves, mrX_moves)

        # this is else_1
        else:
            mrX_moves = [move for move in generate_moves(self.mrX_pos) if move not in self.agents_pos]
            agents_moves = [self.agents_pos]

        # Part of code for Experiment 1 -- needed to be placed before else_1
        """
        elif agents_mcts_type:
            mrX_moves = generate_moves(self.mrX_pos)

            move = self.act_move + 1
            while move % last_seen_length != 0:
                old = mrX_moves
                mrX_moves = []
                for mov in old:
                    for mo in generate_moves(mov):
                        if mo not in mrX_moves:
                            mrX_moves.append(mo)

                move -= 1

            agents_moves = [self.agents_pos]
        """
        # End of part of code for exp. 1

        if not agents_moves:
            agents_moves = generate_agent_moves(self.agents_pos)

        for agents_move in agents_moves:
            for mrX_move in mrX_moves:
                node = Node(mrX_move, agents_move, self.act_move if not self.agents_move else self.act_move + 1,
                            not self.agents_move)
                node.parent = self
                self.leaves.append(node)

    def best(self):
        """
        Find best next move(s) of Mr. X / Agents based on number of visits\n
        :return:                Node with best move of agents / Mr. X
        """
        if not self.leaves:
            return self

        best_leaf = self.leaves[0]
        for leaf in self.leaves:
            if leaf.visited > best_leaf.visited:
                best_leaf = leaf

        return best_leaf

    def print_tree(self, level, tree_level):
        """
        Prints the tree using recursion of MCTS method for specified tree_level\n
        :param level:           Actual level of print
        :param tree_level:      Maximal wanted level of print
        :return:                None
        """
        if self.visited > 0:
            print('  ' * level + '|__ M:', self.act_move + 1, self.mrX_pos, ' ', self.agents_pos, 'wins: ', self.wins,
                  'value: ', self.value, 'visited:', self.visited, 'UCB: ', self.UCB)
        if tree_level <= level:
            return
        else:
            for leaf in self.leaves:
                leaf.print_tree(level + 1, tree_level)


def simulate(node):
    """
    Makes full simulation from choosed Node until last round of the game and return its result from Agends prespection\n
    :param node:                Choosed Node, where should simulation start
    :return:                    True when win Agents and False when win Mr. X
    """

    if node.mrX_pos in node.agents_pos:
        return (True if agents_mcts_type else False), 9

    while node.act_move < max_moves:
        if node.agents_move:
            mrX_moves = [move for move in generate_moves(node.mrX_pos) if move not in node.agents_pos]

            if not mrX_moves:
                return (True if agents_mcts_type else False), 0
                # before FINAL version also used this, but tests showed it is not better:
                # return (True if agents_move else False), (max_moves - node.act_move) / max_moves / 10

            node = Node(rnd.choice(mrX_moves), node.agents_pos, node.act_move + 1, False)
        else:
            agents_moves = generate_agent_moves(node.agents_pos)
            agents_pos = rnd.choice(agents_moves)

            if node.act_move % last_seen_length == 0:
                for moves in agents_moves:
                    if node.mrX_pos in moves:
                        agents_pos = moves

            if node.mrX_pos in agents_pos:
                return (True if agents_mcts_type else False), 0
                # before FINAL version also used this, but tests showed it is not better:
                # return (True if agents_move else False), (max_moves - node.act_move) / max_moves / 10

            if not [move for move in generate_moves(node.mrX_pos) if move not in agents_pos]:
                return (True if agents_mcts_type else False), 0
                # before FINAL version also used this, but tests showed it is not better:
                # return (True if agents_move else False), (max_moves - node.act_move) / max_moves / 10

            node = Node(node.mrX_pos, agents_pos, node.act_move, True)

    return (False if agents_mcts_type else True), 0


def mc_agents_move(mrX_ls_pos, last_seen_len_in, agents_pos, act_move, i_time, tree_level, max_moves_in, field_size_in):
    """
    Makes one move of all agents using Monte Carlo Tree Search method\n
    :param mrX_ls_pos:          Last known position of Mr. X
    :param last_seen_len_in:    Number of rounds, when Mr. X shows up
    :param agents_pos:          Position of all agents
    :param act_move:            Actual number of move (round)
    :param i_time:              Time in secs for iterations for MCTS
    :param tree_level:          Max level to print MCTS tree
    :param max_moves_in:        Max of moves (rounds) to end of game
    :param field_size_in:       Size of field
    :return:                    Next moves (positions) of all agents
    """
    # Setting values of Global variables
    global field_size, last_seen_length, max_moves
    field_size = field_size_in
    last_seen_length = last_seen_len_in
    max_moves = max_moves_in

    # First 2 moves, agents dont have clue, where mrX is, so moves randomly -- Unused in experiment 2!
    if mrX_ls_pos == [-1, -1]:
        return rnd.choice(generate_agent_moves(agents_pos))

    # Create root node
    root = Node(mrX_ls_pos, agents_pos, act_move, False, 1)

    # Part of code for Experiment 2
    """
    best_positions = [[1, 1], [field_size - 2, field_size - 2], [1, field_size - 2], [field_size - 2, 1]]
    if act_move == 0:
        middle = [field_size // 2, field_size // 2]
        if middle not in agents_pos:
            root = Node(middle, agents_pos, act_move, False, 1)
        else:
            root = Node(rnd.choice([pos for pos in best_positions if pos not in agents_pos]), agents_pos, act_move,
                        False, 1)
    elif act_move == 1:
        root = Node(rnd.choice([pos for pos in best_positions if pos not in agents_pos]), agents_pos, act_move, False,
                    1)
    """
    # End of part of code for exp. 2

    # Iterate for number of iterations until the iter time set at the start of program
    end_time = time.time() + i_time
    while time.time() < end_time:

        # Selection and Expansion
        leaf = root.select_leaf()

        # Simulation
        win, value = simulate(leaf)

        # Backpropagation
        while True:
            if win:
                leaf.wins += 1 + value

            leaf.visited += 1

            if leaf.parent is None:
                break

            leaf = leaf.parent

    # Selection of the Node with best move for agents
    best_child = root.best()

    # Prints MCTS tree when its set from start of program
    if tree_level >= 0:
        root.print_tree(0, tree_level)
        print('Chosen:', best_child.agents_pos)

    return best_child.agents_pos


def mc_mrX_move(mrX_pos, last_seen_len_in, agents_pos, act_move, i_time, tree_level, max_moves_in, field_size_in):
    """
    Makes one move of Mr. X using Monte Carlo Tree Search method\n
    :param mrX_pos:             Actual position of Mr. X
    :param last_seen_len_in:    Number of rounds, when Mr. X shows up
    :param agents_pos:          Position of all agents
    :param act_move:            Actual number of move (round)
    :param i_time:              Time for iterations for MCTS
    :param tree_level:          Max level to print MCTS tree
    :param max_moves_in:        Max of moves (rounds) to end of game
    :param field_size_in:       Size of field
    :return:                    Next move (position) of Mr. X
    """
    # Setting values of Global variables
    global field_size, max_moves, agents_mcts_type, last_seen_length
    field_size = field_size_in
    max_moves = max_moves_in
    agents_mcts_type = False
    last_seen_length = last_seen_len_in

    # Create root node
    root = Node(mrX_pos, agents_pos, act_move - 1, True, 1)

    # Iterate for number of iterations until the iter time set at the start of program
    end_time = time.time() + i_time
    while time.time() < end_time:

        # Selection with Expansion
        leaf = root.select_leaf()

        # Simulation
        win, _ = simulate(leaf)

        # Backpropagation
        while True:
            if win:
                leaf.wins += 1

            leaf.visited += 1

            if leaf.parent is None:
                break

            leaf = leaf.parent

    # Selection of the Node with best move for agents
    best_child = root.best()

    if tree_level >= 0:
        root.print_tree(0, tree_level)
        print('Chosen:', best_child.mrX_pos)

    return best_child.mrX_pos


# Help functions ------------------------------------------------------------------------------------------------------v
def get_agents_direction_moves(agents_position, all_agents_moves, all_mrX_moves):
    """
    Gets moves of agents, that has right direction to all possible Mr. X positions\n
    :param agents_position:     Current agents position
    :param all_agents_moves:    All agents moves
    :param all_mrX_moves:       All Mr. X possible moves
    :return:                    All agent moves in directions to Mr. X possible positions
    """

    # This function was added into the final version
    ret_moves = []
    for agents_move in all_agents_moves:
        for mrX_move in all_mrX_moves:
            add = True
            for i in range(len(agents_move)):
                if (abs(agents_move[i][0] - mrX_move[0]) + abs(agents_move[i][1] - mrX_move[1])) > \
                        (abs(agents_position[i][0] - mrX_move[0]) + abs(agents_position[i][1] - mrX_move[1])):
                    add = False
            if add and agents_move not in ret_moves:
                ret_moves.append(agents_move)

    return ret_moves


def generate_agent_moves(agents):
    """
    Function for generating set of all possible moves for agents\n
    :param agents:              Current positions of agents
    :return:                    List of possible moves for agents
    """
    moves = []
    for agent in agents:
        moves.append([move for move in generate_moves(agent) if move not in agents[agents.index(agent):]])

    # Source: https://docs.python.org/3/library/itertools.html#itertools.product
    return [move for move in list(itertools.product(*moves)) if not contain_duplicates(move)]


def generate_moves(pos):
    """
    Generates all possible moves, that can be made from gived current possition\n
    :param pos:                 Current position for generating
    :return:                    All possible moves (next possitions)
    """
    moves = [[pos[0] + 1, pos[1]], [pos[0] - 1, pos[1]], [pos[0], pos[1] + 1], [pos[0], pos[1] - 1]]
    return [move for move in moves if (0 <= move[0] < field_size) and (0 <= move[1] < field_size)]


def contain_duplicates(items):
    """
    Check if gived list contains duplicate items\n
    :param items:               List of items (possitions in this situation)
    :return:                    True when there are some duplicate items in list, False otherwise
    """
    for item in items:
        if items.count(item) > 1:
            return True
    return False

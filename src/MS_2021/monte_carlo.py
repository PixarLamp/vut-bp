# author: Michal Sova
# year: 2021
# file: monte_carlo.py
# description: File with class and functions needed for Monte Carlo Tree Search

import random
import math
import time
import itertools


class Node:
    def __init__(self, mrx, agents, move_counter, moves, field_size):
        """
        :param mrx: position of mr X         mrx = [x, y]
        :param agents: position of agents    agents = [[x, y], ...]
        :param move_counter: number of moves played
        :param moves: how many moves are in the game (length of the game)
        :param field_size: size of the game field
        """
        self.parent = None
        self.leaves = list()
        self.UCB = float('inf')  # UCB = value + C * sqrt(log(n_of_simulations)/visited)
        self.visited = 0
        self.value = 0
        self.agents = []
        for agent in agents:
            self.agents.append(agent)
        self.mrx_pos = mrx
        self.move_counter = move_counter
        self.total_moves = moves
        self.field_size = field_size

    def random_child(self):
        """ random move
        :returns: random move of agents and mr X
        :returns: None if mrX has no valid moves
        """
        # random move of MrX
        mrx_moves = generate_moves(self.mrx_pos, self.field_size)
        for move in mrx_moves:
            for agent in self.agents:
                if move == agent:
                    mrx_moves.remove(move)
                    break
        if len(mrx_moves) == 0:
            return None
        rnd_mrx = random.randint(0, len(mrx_moves) - 1)

        # random moves of agents
        agent_moves = generate_agent_moves(self.agents, self.field_size)
        rnd_a = random.randint(0, len(agent_moves) - 1)
        agents_move = []
        for a in range(len(self.agents)):
            agents_move.append(agent_moves[rnd_a][a])
        return Node(mrx_moves[rnd_mrx], agents_move, self.move_counter + 1, self.total_moves, self.field_size)

    def expand(self):
        """
        expand tree with possible moves of agents
        mr X move is random
        """
        if self.move_counter >= self.total_moves:
            return
        mrx_moves = generate_moves(self.mrx_pos, self.field_size)
        agent_moves = generate_agent_moves(self.agents, self.field_size)
        for move in agent_moves:
            for mrx_move in mrx_moves:
                node = Node(mrx_move, move, self.move_counter + 1, self.total_moves, self.field_size)
                node.parent = self
                self.leaves.append(node)

    def traversal(self, n_of_simulations):
        """
        traverse through tree, find leaf with best UCB value
        :param n_of_simulations: number of simulations so far
        :return: unvisited leaf node or node with best UCB
        """
        # is it leaf node
        if len(self.leaves) == 0:
            # has not been visited
            if self.visited == 0:
                return self
            # has been visited -> expand and return first leaf
            else:
                self.expand()
                if len(self.leaves) == 0:
                    return self
                else:
                    return self.leaves[0]

        if len(self.leaves) == 0:
            return self
        # traverse tree(traverse to leaf with best UCB)
        best_leaf = self.leaves[0]
        for leaf in self.leaves:
            if leaf.visited == 0:  # division by zero : ln(t/n), where n = leaf.visited
                return leaf

            # exploration/exploitation: chosen leaf with best UCB
            # UCB = value + C * sqrt(ln(t/n)), where:
            #       value - average reward/value of all nodes beneath this node
            #       C - well chosen constant
            #       t - total number of simulations
            #       n - number of times child node has been visited
            leaf.UCB = leaf.value + 100.0 * math.sqrt(math.log(n_of_simulations) / leaf.visited)
            if leaf.UCB > best_leaf.UCB:
                best_leaf = leaf

        return best_leaf.traversal(n_of_simulations)

    def best(self):
        """
        find best next move based on number of visits
        :return: best node with move of agents
        """
        best_leaf = self.leaves[0]
        for leaf in self.leaves:
            if leaf.visited > best_leaf.visited:
                best_leaf = leaf

        return best_leaf

    def print_tree(self, level, tree_level):
        print(" |" * level + "--", self.mrx_pos, ' ', self.agents, "value:", int(self.value), "visited:", self.visited)
        if tree_level <= level:
            return
        else:
            for leaf in self.leaves:
                leaf.print_tree(level + 1, tree_level)


def simulate(node):
    """execute simulation with random moves from given node
    :param node: starting node (position)
    :returns value of result
    """
    retval = 0
    while True:
        node_next = node.random_child()

        # no valid moves for mr X -> next move of mr X will be on one of the agents
        if node_next is None:
            return retval + (100 - node.move_counter) * 50

        # one of agents caught mr X
        for agent in node_next.agents:
            if node.mrx_pos == agent:
                return retval + (100 - node.move_counter) * 50

        # mr X escaped
        if node_next.move_counter >= node.total_moves:
            return retval - 100

        for agent in node_next.agents:
            dist = distance_of(agent, node.mrx_pos)
            if dist > node.total_moves - node.move_counter:
                retval -= (dist + 10) * 5
            else:
                retval -= dist

        # go to next move
        node = node_next


def simulate_mrx(node):
    """execute simulation with random moves from given node
    :param node: starting node (position)
    :returns value of result
    """
    retval = 0
    while True:

        # no valid moves for mr X -> next move of mr X will be on one of the agents
        if node is None:
            return retval - 100

        # one of agents caught mr X
        for agent in node.agents:
            if node.mrx_pos == agent:
                return retval - (100 + node.move_counter) * 50

        # mr X escaped
        if node.move_counter >= node.total_moves:
            return retval + 100

        for agent in node.agents:
            dist = distance_of(agent, node.mrx_pos)
            if dist > node.total_moves - node.move_counter:
                retval += (dist + 10) * 5
            else:
                retval += dist

        # go to next move
        node = node.random_child()


def mc_agents_move(last_seen, mrx_ls_pos, last_seen_length, agents, move_counter, mc_time, tree_level, moves, field_size):
    """function returns new positions of agents using monte-carlo algorithm
    :param last_seen:         number of moves, mr X was last seen
    :param mrx_ls_pos:        last position of mr X mrx_ls_pos = [x, y]
    :param last_seen_length:  every last_seen_length will agents see mr X
    :param agents:            position of agents   agents = [[Xa1, Ya1], [Xa2, Ya2], ...]
    :param move_counter:      how many moves has been played
    :param mc_time:           time assigned to run monte-carlo simulation (in seconds)
    :param tree_level:        level of decision tree
    :param moves:             how many moves are in the game (length of the game)
    :param field_size:        size of the game
    :returns          new position of agents
    """

    if mrx_ls_pos[0] == -1:
        agent_moves = generate_agent_moves(agents, field_size)
        rnd_a = random.randint(0, len(agent_moves) - 1)
        return agent_moves[rnd_a]

    # calculate root node
    root = Node(mrx_ls_pos, agents, move_counter, moves, field_size)
    for i in range(last_seen % last_seen_length):  # when was mrX last seen (0 - last_seen_length)
        mrx_moves = generate_moves(root.mrx_pos, field_size)
        root = Node(mrx_moves[random.randint(0, len(mrx_moves)) - 1], agents, move_counter, moves, field_size)

    root.expand()

    number_of_simulations = 0
    end_time = time.time() + mc_time
    while time.time() < end_time:
        # selection/expansion
        leaf = root.traversal(number_of_simulations)

        # rollout/simulation
        value = simulate(leaf)
        number_of_simulations += 1

        # backpropagation
        while True:
            leaf.value += value
            leaf.visited += 1
            if leaf.parent is None:
                break
            leaf = leaf.parent

    best_child = root.best()
    if tree_level >= 0:
        root.print_tree(0, tree_level)
        print('chosen:', best_child.agents)
    return best_child.agents


def mc_mrx_move(mrx_pos, agents, move_counter, mc_time, tree_level, moves, field_size):
    """function returns new position of mrX using monte-carlo algorithm
    :param mrx_pos: current position of mrX, mrx_pos = [x, y]
    :param agents: current position of agents, agents = [[Xa1, Ya1], [Xa2, Ya2], ...]
    :param move_counter: how many moves has been played
    :param mc_time: time assigned to run monte-carlo simulation (in seconds)
    :param tree_level: level of decision tree
    :param moves: how many moves are in the game (length of the game)
    :param field_size: size of the game
    :returns new position of mr X
    """
    root = Node(mrx_pos, agents, move_counter, moves, field_size)
    root.expand()
    number_of_simulations = 0
    end_time = time.time() + mc_time
    while time.time() < end_time:
        # selection/expansion
        leaf = root.traversal(number_of_simulations)

        # rollout/simulation
        value = simulate_mrx(leaf)
        number_of_simulations += 1

        # backpropagation
        while True:
            leaf.value += value
            leaf.visited += 1
            if leaf.parent is None:
                break
            leaf = leaf.parent

    best_child = root.best()

    if tree_level >= 0:
        root.print_tree(0, tree_level)
        print('chosen:', best_child.mrx_pos)

    return best_child.mrx_pos


def generate_agent_moves(agents, field_size):
    """ generate set of possible moves for agents
    :param agents: current position of agents agents = [[x, y], [...], ...]
    :param field_size: size of game
    :returns: list of possible moves for agents [[[Xa1, Ya1], [Xa2, Ya2], ..., [Xan, Yan]], ...]
    """
    tmp_moves = []
    for agent in agents:
        a_moves = generate_moves(agent, field_size)
        for a_move in a_moves:
            if a_move in agents[agents.index(agent):]:
                a_moves.remove(a_move)
        tmp_moves.append(a_moves)

    agents_moves = []
    for items in itertools.product(*tmp_moves):
        if not has_duplicate(items):
            agents_moves.append(list(items))

    return agents_moves


# https://thispointer.com/python-3-ways-to-check-if-there-are-duplicates-in-a-list/
def has_duplicate(items):
    """ checks if given list has duplicates
    :param items: list or tuple
    :returns: True, if given list has duplicates
    :returns: False otherwise
    """
    for elem in items:
        if items.count(elem) > 1:
            return True
    return False


def distance_of(a, b):
    """returns distance between two points in number of moves
    (i.e. how many in-game moves would it take from point 'a' to point 'b')
    :param a: first point a = [ax, ay]
    :param b: second point b = [bx, by]
    :return: distance between two points
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def generate_moves(apos, field_size):
    """ generate set of possible moves from current position
    :param apos: current position apos = [x, y]
    :param field_size: size of game field
    :returns: list of possible moves [[x1, y1], [x2, y2], ...]
    """
    moves = list()
    amove = [apos[0] + 1, apos[1]]
    if amove[0] < field_size:
        moves.append(amove)

    amove = [apos[0] - 1, apos[1]]
    if 0 <= amove[0]:
        moves.append(amove)

    amove = [apos[0], apos[1] + 1]
    if amove[1] < field_size:
        moves.append(amove)

    amove = [apos[0], apos[1] - 1]
    if 0 <= amove[1]:
        moves.append(amove)

    return moves


def random_agent_moves(agents, fs):
    """ function returns random set of moves for agents
    :param agents: current position of agents
    :param fs: size of field
    :return: list of random moves for each agent [[a1x, a1y], [a2x, a2y], ...]
    """
    moves = generate_agent_moves(agents, fs)
    return moves[random.randint(0, len(moves) - 1)]


def random_mrx_moves(mrx, fs):
    """ functon retruns random move for mr X
    :param mrx: current position of mr X
    :param fs: size of field
    :return: new position of mr X [x, y]
    """
    moves = generate_moves(mrx, fs)
    return moves[random.randint(0, len(moves) - 1)]

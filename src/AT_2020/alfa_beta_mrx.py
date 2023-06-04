# author: Adrián Tulušák
# year: 2020
# file: alfa_beta_mrx.py
# description: file contains all needed methods for running of alpha-beta algorithm by Mr. X

from AT_2020 import alfabeta_node as nd, environment as env


class AlfaBeta:
    def __init__(self, field_size=5):
        self.root = nd.Node()
        self.field_size = field_size

    def move_mrx(self, agents, mrx):
        if env.get_valid_moves_mrx(mrx, agents, fs=self.field_size):

            self.explore_state_space(agents, mrx)

            new_move_node = self.root.best_way

            return new_move_node.mrx
        else:
            return -1

    def explore_state_space(self, agents, mrx):
        self.root.reset()
        for agent in agents:
            self.root.agents.append(agent)
        self.root.mrx = mrx

        self.make_alfabeta_step(self.root)

    def make_alfabeta_step(self, node):
        if node.depth > 2:
            return self.evaluate_state(node)
        elif node.turn == "A":
            return self.player_A(node)
        elif node.turn == "B":
            return self.player_B(node)
        else:
            raise Exception(SystemError)

    # player A = Mr.X
    def player_A(self, node):
        new_valid_locations = env.get_valid_moves_mrx(node.mrx, node.agents, fs=self.field_size)

        if new_valid_locations == []:
            # print("ERR: empty moves")
            return self.evaluate_state(node)

        for new_location in new_valid_locations:
            if node.alfa < node.beta:
                new_node = nd.Node(node.agents, new_location, parrent=node, alfa=node.alfa, beta=node.beta,
                                   depth=node.depth,
                                   turn="B")
                node.children.append(new_node)

                node_result = self.make_alfabeta_step(node.children[-1])

                if node_result > node.alfa:
                    node.alfa = node_result
                    node.best_way = node.children[-1]

            else:
                break

        return node.alfa

    # player B = agents
    def player_B(self, node):
        new_valid_locations = env.get_valid_moves_agents(node.agents, fs=self.field_size)

        if new_valid_locations == []:  # use only when play only AI against itself
            print("ERR: empty moves2")
            return self.evaluate_state(node)

        for new_locations in new_valid_locations:
            if node.alfa < node.beta:
                new_node = nd.Node(new_locations, node.mrx, parrent=node, alfa=node.alfa, beta=node.beta,
                                   depth=node.depth + 1, turn="A")
                node.children.append(new_node)

                node_result = self.make_alfabeta_step(node.children[-1])

                if node_result < node.beta:
                    node.beta = node_result
                    node.best_way = node.children[-1]

            else:
                break

        return node.beta

    def evaluate_state(self, evaluated_node):
        finished = False
        node = evaluated_node
        value = 0
        while not finished:
            if node == self.root:
                finished = True

            """
            # for every posible catch during this branch is state
            if node.mrx == node.agents[0]:
                value -= (3 - node.depth) * 50
                value += self.evaluate_distance(node.mrx, node.agents[1]) * (3 - node.depth)
            elif node.mrx == node.agents[1]:
                value -= (3 - node.depth) * 50
                value += self.evaluate_distance(node.mrx, node.agents[0]) * (3 - node.depth)
            else:
                value1 = self.evaluate_distance(node.mrx, node.agents[0]) * (3 - node.depth)
                value2 = self.evaluate_distance(node.mrx, node.agents[1]) * (3 - node.depth)
                value += min(value1, value2)
                value += max(value1, value2) // 2
            """

            if node.mrx in node.agents:
                value -= (3 - node.depth) * 50
                val = 0
                for agent in node.agents:
                    if agent != node.mrx:
                        val += self.evaluate_distance(node.mrx, agent) * (3 - node.depth)
                value += val // (len(node.agents) - 1)
            else:
                values = []
                for agent in node.agents:
                    values.append(self.evaluate_distance(node.mrx, agent) * (3 - node.depth))

                value += min(values)
                value += max(values) // len(values)

            node = node.parrent

        return value

    def evaluate_distance(self, pos1, pos2):
        p1x = pos1 % self.field_size
        p1y = pos1 // self.field_size

        p2x = pos2 % self.field_size
        p2y = pos2 // self.field_size

        distance = abs(p1x - p2x) + abs(p1y - p2y)

        return distance

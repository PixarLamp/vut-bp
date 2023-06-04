# author: Adrián Tulušák
# year: 2020
# file: alfa_beta.py
# description: file contains all needed methods for running of alpha-beta algorithm

from AT_2020 import alfabeta_node as nd, environment as env


class AlfaBeta:
    def __init__(self, fs=5):
        self.root = nd.Node()
        self.field_size = fs

    def choose_new_move_agents(self):
        new_move_node = self.root.best_way

        self.root = new_move_node
        new_agents = self.root.agents

        ###############################################
        # MOVE WITH MR.X IN ALFA-BETA TREE ############
        new_move_node = self.root.best_way          ###
                                                    ###
        if new_move_node == False:                  ###
            raise Exception(SystemError)            ###
                                                    ###
        self.root = new_move_node                   ###
        ###############################################

        return new_agents

    def explore_state_space(self, agents, mrx):
        self.root = nd.Node()
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

    def player_A(self, node):
        new_valid_locations = env.get_valid_moves_agents(node.agents, self.field_size)

        if not new_valid_locations:
            return self.evaluate_state(node)

        for new_location in new_valid_locations:
            if node.alfa < node.beta:
                new_node = nd.Node(new_location, node.mrx, parrent=node, alfa=node.alfa,
                                   beta=node.beta, depth=node.depth, turn="B")
                node.children.append(new_node)

                node_result = self.make_alfabeta_step(node.children[-1])

                if node_result > node.alfa:
                    node.alfa = node_result
                    node.best_way = node.children[-1]

            else:
                break

        return node.alfa

    def player_B(self, node):
        # new_valid_locations = env.get_valid_moves_mrx_vs_player(node.mrx)  # variant for playing with player #
        new_valid_locations = env.get_valid_moves_mrx(node.mrx, node.agents, fs=self.field_size)  # use only when play only AI against itself

        if not new_valid_locations:  # use only when play only AI against itself
            # print("ERR: empty moves")
            return self.evaluate_state(node)

        for new_location in new_valid_locations:
            if node.alfa < node.beta:
                new_node = nd.Node(node.agents, new_location, parrent=node, alfa=node.alfa, beta=node.beta,
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

            # for every posible chatch during this branch is state

            # V.1
            if node.mrx in node.agents:
                value += (3 - node.depth) * 50
            else:
                for agent in node.agents:
                    value -= self.evaluate_distance(node.mrx, agent) * node.depth

            # V.2
            # if node.mrx == node.a1:
            #     value += (3 - node.depth) * 50
            #     value -= self.evaluate_distance(node.mrx, node.a2) * node.depth
            # elif node.mrx == node.a2:
            #     value += (3 - node.depth) * 50
            #     value -= self.evaluate_distance(node.mrx, node.a1) * node.depth
            # else:
            #     value1 = self.evaluate_distance(node.mrx, node.a1) * node.depth
            #     value2 = self.evaluate_distance(node.mrx, node.a2) * node.depth
            #     value -= min(value1, value2)

            # V.3
            # if node.mrx == node.a1:
            #     value += (3 - node.depth) * 50
            #     value -= self.evaluate_distance(node.mrx, node.a2)
            # elif node.mrx == node.a2:
            #     value += (3 - node.depth) * 50
            #     value -= self.evaluate_distance(node.mrx, node.a1)
            # else:
            #     value1 = self.evaluate_distance(node.mrx, node.a1)
            #     value2 = self.evaluate_distance(node.mrx, node.a2)
            #     value -= min(value1, value2)

            # V.4 ###############
            # if node.mrx == node.a1:
            #     value += (3 - node.depth) * 50
            #     value -= self.evaluate_distance(node.mrx, node.a2)
            # elif node.mrx == node.a2:
            #     value += (3 - node.depth) * 50
            #     value -= self.evaluate_distance(node.mrx, node.a1)
            # else:
            #     value1 = self.evaluate_distance(node.mrx, node.a1)
            #     value2 = self.evaluate_distance(node.mrx, node.a2)
            #     value -= min(value1, value2)
            #     value -= max(value1, value2) // 2

            node = node.parrent

        return value

    def evaluate_distance(self, pos1, pos2):
        p1x = pos1 % 5
        p1y = pos1 // 5

        p2x = pos2 % 5
        p2y = pos2 // 5

        distance = abs(p1x - p2x) + abs(p1y - p2y)

        return distance

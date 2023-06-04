# author: Adrián Tulušák
# year: 2020
# file: alfa_beta_node.py
# description: file contains class Node that is used in alpha-beta algorithm

class Node:
    def __init__(self, agents=None, mrx=0, parrent=None, alfa=-9999, beta=9999, depth=0, turn="A"):
        if agents is None:
            agents = [0, 0]
        self.parrent = parrent  # reference to parrent node
        self.alfa = alfa  # alfa value
        self.beta = beta  # beta value
        self.best_way = None  # child node with best way
        self.depth = depth  # depth in exploration/in tree
        self.turn = turn  # player A/player B
        self.children = []  # list of children nodes
        self.agents = []
        for agent in agents:
            self.agents.append(agent)
        self.mrx = mrx  # position of mrX in this node
        self.number = 0  # order number of node

    def reset(self, turn="A"):
        self.parrent = None
        self.alfa = -9999
        self.beta = 9999
        self.best_way = None
        self.depth = 0
        self.turn = turn
        self.children = []
        self.agents.clear()
        self.mrx = 0

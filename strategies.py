from networkx import nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
import operator
import multiprocessing as mp
import torch

def sigmoid(x): return float(1) / float((1 + math.exp(-x)))

class Agent():
    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.type = None
        self.futureColor = 'white'
        self.currentColor = 'white'
        self.neighbors = None
        self.visibleNeighbors = None
        self.invisibleNeighbors = None
        self.countColorChanges = 0
        self.timeColorChange = []


    def instantiateNeighbors(self, agents):
        """""
        This function generate the agent's neighbors and assign them by their type(vis / inv)
        """
        neighbor_list = []
        visible_list = []
        invisible_list = []
        for agent in agents:
            if agent.unique_id in self.neighbors:
                neighbor_list.append(agent)
                if agent.type == 'vis':
                    visible_list.append(agent)
                elif agent.type == 'reg' or agent.type == 'adv':
                    invisible_list.append(agent)
        self.neighbors = neighbor_list
        self.visibleNeighbors = visible_list
        self.invisibleNeighbors = invisible_list

    def setParametrs(self, indicator):
        """""
        This function set the parameters which will multiply with the coefficients from T1, T2
        indicator : In case of picking initial color : indicator = True
                    In case of changing color : indicator = False
        """
        p_list = [1]
        inv_neighbors = self.invisibleNeighbors
        vis_neighbors = self.visibleNeighbors
        GLV = 0  # Green Local Vis
        RLV = 0  # Red Local Vis
        GLI = 0  # Green Local Inv
        RLI = 0  # Red Local Inv
        if indicator:  # Picking initial color
            if len(vis_neighbors) != 0:
                GLV = float(len([neighbor for neighbor in vis_neighbors if neighbor.currentColor == 'green'])) \
                                  / float(len(vis_neighbors))
                RLV = float(len([neighbor for neighbor in vis_neighbors if neighbor.currentColor == 'red']))\
                                / float(len(vis_neighbors))
            if len(inv_neighbors) != 0:
                GLI = float(len([neighbor for neighbor in inv_neighbors if neighbor.currentColor == "green"]))\
                                  / float(len(inv_neighbors))
                RLI = float(len([neighbor for neighbor in inv_neighbors if neighbor.currentColor == "red"]))\
                                / float(len(inv_neighbors))
            diff_inv = abs(RLI - GLI)
            diff_vis = abs(RLV - GLV)
            p_list.extend([diff_inv, diff_vis, len(inv_neighbors), len(vis_neighbors), 1, GLI, GLV, RLI, RLV])
        else:  # changing color
            OLV = 0  # Opposite Local Vis
            CLV = 0  # Current Local Vis
            OLI = 0  # Opposite Local Inv
            CLI = 0  # Current Local Inv
            if len(inv_neighbors) != 0:
                opposite_color_inv_neighbors = float(len([neighbor for neighbor in inv_neighbors
                                                          if
                                                          neighbor.currentColor != "white" and neighbor.currentColor != self.currentColor]))
                current_color_inv_neighbors = float(len([neighbor for neighbor in inv_neighbors
                                                         if
                                                         neighbor.currentColor != "white" and neighbor.currentColor == self.currentColor]))
                OLI = opposite_color_inv_neighbors / float(len(inv_neighbors))
                CLI = current_color_inv_neighbors / float(len(inv_neighbors))
            if len(vis_neighbors) != 0:
                opposite_color_vis_neighbors = float(len([neighbor for neighbor in vis_neighbors
                                                          if
                                                          neighbor.currentColor != "white" and neighbor.currentColor != self.currentColor]))
                current_color_vis_neighbors = float(len([neighbor for neighbor in vis_neighbors
                                                         if
                                                         neighbor.currentColor != "white" and neighbor.currentColor == self.currentColor]))
                OLV = opposite_color_vis_neighbors / float(len(vis_neighbors))
                CLV = current_color_vis_neighbors / float(len(vis_neighbors))
            p_list.extend([OLI, OLV, CLI, CLV, len(inv_neighbors), len(vis_neighbors)])
        return p_list

    def setProbabilty(self, PL, T1, T2, T3, indexline, indicator):
        """""
        This function return:
            In case of picking initial color : {p(red): x, p(green): y}
            In case of changing color : p(change)
        PL : Parameters List;
             picking initial color : PL = [1, diff_inv, diff_vis, inv_neighbors, vis_neighbors, 1, GLI, GLV, RLI, RLV]
             changing color : PL = [1, OLI, OLV, CLI, CLV, inv_neighbors, vis_neighbors]
        T1 : Color Picking Model table
        T2 : Red Picking Model table
        T3 : Color Changing Model table
        indexline : The line index in T1/T2/T3 which need to be used(depend on the case)
                    Example : if the node is reg and have no vis neighbors, indexline = 0
        indicator : In case of picking initial color : indicator = True
                    In case of changing color : indicator = False
        """
        function = 0
        if indicator:  # Picking initial color
            colorpicking_parameters = PL[:5]
            x = T1[indexline]
            for i in range(len(x)):
                function += x[i] * colorpicking_parameters[i]
            p_choose = sigmoid(function)
            function = 0
            redpicking_parameters = PL[5:]
            x = T2[indexline]
            for i in range(len(x)):
                function += x[i] * redpicking_parameters[i]
            p_red = sigmoid(function)
            p_green = 1 - p_red
            return {'red': p_choose * p_red, 'green': p_choose * p_green}
        else:  # changing color
            x = T3[indexline]
            for i in range(len(x)):
                function += x[i] * PL[i]
                p_change = sigmoid(function)
            return p_change



    def chooseColor(self, p, indicator):
        """""
        This function set the color of node in the graph
        P : Can be either a dict; {p(red): x, p(green): y} or a number; p(change)
        indicator : In case of picking initial color : indicator = True
        In case of changing color : indicator = False
        """
        if indicator:  # Picking initial color
            if random.random() < p['red']:
                self.countColorChanges += 1
                self.futureColor = 'red'
            elif random.random() < p['green']:
                self.countColorChanges += 1
                self.futureColor = 'green'
        else:  # changing color
            if random.random() < p:
                self.countColorChanges += 1
                if self.currentColor == 'green':
                    self.futureColor = 'red'
                else:
                    self.futureColor = 'green'

    def colorsState(self, circles):
        count_colors = {'green': 0, 'red': 0}
        for neighbor in self.neighbors:
            if neighbor.currentColor == 'red':
                count_colors['red'] += 1
            elif neighbor.currentColor == 'green':
                count_colors['green'] += 1
            if circles == 2:
                for nneighbor in neighbor.neighbors:
                    if nneighbor.currentColor == 'red':
                        count_colors['red'] += 1
                    elif nneighbor.currentColor == 'green':
                        count_colors['green'] += 1
        return count_colors

    def pickDominantColor(self, circles):
        dominant_color = (max(self.colorsState(circles).items(), key=operator.itemgetter(1))[0])
        self.futureColor = dominant_color

    def expectedConsensus(self, agnts):
        count_colors = {'green': 0, 'red': 0}
        for agent in agnts:
            if agent.currentColor == 'red':
                count_colors['red'] += 1
            elif agent.currentColor == 'green':
                count_colors['green'] += 1
        return max(count_colors.values()) / len(agnts)


    def pickInitialColor(self, T1, T2):
        """""
        This function implementing the process of picking initial color in case the node's current color is white
        T1 : Color Picking Model table
        T2 : Red Picking Model table
        parameters : A list of parameters which will multiply with the coefficients from T1, T2
        prob_pair : {p(red): x, p(green): y}
        """
        parameters = self.setParametrs(True)
        if self.type == 'reg':
            if len(self.visibleNeighbors) == 0:
                prob_pair = self.setProbabilty(parameters, T1, T2, None, 0, True)
            else:
                prob_pair = self.setProbabilty(parameters, T1, T2, None, 1, True)
        elif self.type == 'vis':
            if len(self.visibleNeighbors) == 0:
                prob_pair = self.setProbabilty(parameters, T1, T2, None, 2, True)
            else:
                prob_pair = self.setProbabilty(parameters, T1, T2, None, 3, True)
        else:
            if len(self.visibleNeighbors) == 0:
                prob_pair = self.setProbabilty(parameters, T1, T2, None, 4, True)
            else:
                prob_pair = self.setProbabilty(parameters, T1, T2, None, 5, True)
        self.chooseColor(prob_pair, True)

    def pickSubsequentColor(self, T3):
        """""
        This function implementing the process of color changing in case the node's current color isn't white
        T3 : Color Changing Model table
        parameters : A list of parameters which will multiply with the coefficients from T3
        prob_of_change : p(change color)
        """
        parameters = self.setParametrs(False)
        if self.type == 'reg':
            if len(self.visibleNeighbors) == 0:
                prob_of_change = self.setProbabilty(parameters, None, None, T3, 0, False)
            else:
                prob_of_change = self.setProbabilty(parameters, None, None, T3, 1, False)
        elif self.type == 'vis':
            if len(self.visibleNeighbors) == 0:
                prob_of_change = self.setProbabilty(parameters, None, None, T3, 2, False)
            else:
                prob_of_change = self.setProbabilty(parameters, None, None, T3, 3, False)
        else:
            if len(self.visibleNeighbors) == 0:
                prob_of_change = self.setProbabilty(parameters, None, None, T3, 4, False)
            else:
                prob_of_change = self.setProbabilty(parameters, None, None, T3, 5, False)
        self.chooseColor(prob_of_change, False)


def readConfigurations(openfile):
    configurations = pd.read_csv(openfile, header=None)
    configurationsList = configurations.iloc[:, :].values.tolist()
    matrix = []
    for line in range(len(configurationsList)):
            configurationsList[line] = configurationsList[line][0].split(' ')
            configurationsList[line].extend(['averagetimeofconsensus:', [], 'consensusrate:', 0])
            matrix.append(configurationsList[line])
    return(matrix)


def AlbertBarabasi(n, m, d):
    while True:
        G = nx.barabasi_albert_graph(n, m, seed=None)
        degrees = dict(G.degree())
        maxDegree = max([item for item in degrees.values()])
        if nx.is_connected(G) and maxDegree <= d:
            break
    adj_dict = nx.convert.to_dict_of_lists(G)
    # nx.draw(G, with_labels=True)
    # plt.show()
    return(adj_dict)


def ErdosRenyi(n, m, d):
    while True:
        G = nx.gnm_random_graph(n, m, seed=None, directed=False)
        degrees = dict(G.degree())
        maxDegree = max([item for item in degrees.values()])
        if nx.is_connected(G) and maxDegree <= d:
            break
    adj_dict = nx.convert.to_dict_of_lists(G)
    return(adj_dict)


def createAgents(adjDict, NA, NV, NR):
    """""
    this function create an agents with adjacencies as function of the graph's type.
    types of agents are random assignment
    NA : Num of Adversarial nodes
    NV : Num of Visible nodes
    NR : Num of Regular nodes
    """
    agents_list = [[] * len(adjDict) for i in range(len(adjDict))]
    temp_list = [NR * ['reg'], NV * ['vis'], NA * ['adv']]
    type_list = []
    for item in temp_list:
        if len(item) != 0:
            for type in item:
                type_list.append(type)
    random.shuffle(type_list)
    for i in range(len(agents_list)):
        agents_list[i] = Agent(i)
        agents_list[i].neighbors = adjDict[i]
        agents_list[i].type = type_list[i]
    """""
    check the accuracy of the random assignments :
    """""
    # a = []
    # for agent in agents_list:
    #     a.append(agent.type)
    # print(a)
    return agents_list


def checkConsensus(agents, NC):
    """""
    This function checking if the consensus is reached by the consensus team
    agents : [agent_1, agent_2,..., agent_n]
    NC : Num of Consensus nodes
    Note: Any observer can watch the consensus's color by print(count_colors)
    """
    count_colors = {'green': 0, 'red': 0}
    for agent in agents:
        if agent.currentColor == 'white':
            return False
        if agent.type != 'adv':
            if agent.currentColor == 'green':
                count_colors['green'] += 1
            else:
                count_colors['red'] += 1
    if count_colors['green'] == NC or count_colors['red'] == NC:
        # print(count_colors)
        return True
    else:
        return False

def consensusState(agnts):
    count_colors = {'green': 0, 'red': 0}
    for agent in agnts:
        if agent.type != 'adv':
            if agent.currentColor == 'green':
                count_colors['green'] += 1
            elif agent.currentColor == 'red':
                count_colors['red'] += 1
    return max(count_colors.values()) / len([agent for agent in agnts if agent.type != 'adv'])



def addRes(agnts, confg, gtime, nc, T1, T2, T3):
    """""
    This function:
    1) Classify the neighbors for each agent cy their type(vis / inv / adv)
    2) Starting the game in terms of color picking / changing.
    3) Once the consensus team reach to a consensus - the game is terminate.
    """
    conrate = dict()
    for agent in agnts:
        agent.instantiateNeighbors(agnts)
    for i in range(gtime):
        for agent in agnts:
            agent.currentColor = agent.futureColor
        for agent in agnts:
            if agent.currentColor == 'white':
                agent.pickInitialColor(T1, T2)
            else:
                agent.pickSubsequentColor(T3)
        conrate[i] = consensusState(agnts)
        if checkConsensus(agnts, nc):
            lists = sorted(conrate.items())
            f1_keys,f1_values= zip(*lists)
            plt.scatter(f1_keys, f1_values, color='red')
            plt.plot(f1_keys, f1_values)
            plt.title('title')
            plt.xlabel('time')
            plt.ylabel('conrate')
            plt.show()
            confg[9].append(i)
            break


def gameAgent(configs):
    """""
    This function implementing the simulations on all configurations
    configs : matrix of all configurations
    circles : the number of adjacency circles in the graph that the visible node's allow to know
    """
    CPM = pd.read_csv('ColorPickingModel.csv').iloc[:, 2:].values  # Color Picking Model table
    RPM = pd.read_csv('RedPickingModel.csv').iloc[:, 2:].values  # Red Picking Model table
    CCM = pd.read_csv('ColorChangingModel.csv').iloc[:, 2:].values  # Color Changing Model table
    for config in configs:
        print(config)
        game_time, simulations, m = 60, 1, 3  # m : Initial edges for every node in the graph
        num_of_con_idx, num_of_vis_idx, num_of_adv_idx, network_idx, times_idx, con_rate_idx = 7, 5, 3, 0, 9, 11
        if config[num_of_con_idx] == '10':
            max_degree = 7
        elif config[num_of_con_idx] == '20':
            max_degree = 15
        else:
            max_degree = 21
        num_of_con = int(config[num_of_con_idx])
        num_of_adv = int(config[num_of_adv_idx])
        num_of_vis = int(config[num_of_vis_idx])
        num_of_reg = num_of_con - num_of_vis
        BA_edges = (num_of_con + num_of_adv - 3) * m
        ERD_edges = BA_edges
        ERS_edges = int(math.ceil(ERD_edges/2.0))
        if config[network_idx] == 'barabasi-albert':
            for i in range(simulations):
                my_graph = AlbertBarabasi(num_of_con + num_of_adv, m, max_degree)
                agents = createAgents(my_graph, num_of_adv, num_of_vis, num_of_reg)
                addRes(agents, config, game_time, num_of_con, CPM, RPM, CCM)
        elif config[network_idx] == 'erdos-renyi-dense':
            for i in range(simulations):
                my_graph = ErdosRenyi(num_of_con + num_of_adv, ERD_edges, max_degree)
                agents = createAgents(my_graph, num_of_adv, num_of_vis, num_of_reg)
                addRes(agents, config, game_time, num_of_con, CPM, RPM, CCM)
        elif config[network_idx] == 'erdos-renyi-sparse':
            for i in range(simulations):
                my_graph = ErdosRenyi((num_of_con + num_of_adv), ERS_edges, max_degree)
                agents = createAgents(my_graph, num_of_adv, num_of_vis, num_of_reg)
                addRes(agents, config, game_time, num_of_con, CPM, RPM, CCM)
        config[11] = float(len(config[times_idx]) / simulations)
        config[9] = np.mean(config[times_idx])
    return configs


"""creating the model"""
configurations = readConfigurations('InitialConfigs.txt')
game_res = gameAgent(configurations)




# """Writing the results into txt file"""
# with open('FullRecognitionRes.txt', 'w') as configs:
#     for row in game_res:
#         row = [str(item) for item in row]
#         configs.write(' '.join(row) + '\n')



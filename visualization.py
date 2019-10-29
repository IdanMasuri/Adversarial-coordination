import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.axes import Axes


def readConfigurations(filename):
    temp_list = []
    with open(filename, 'r') as reader:
        for line in reader.readlines():
            line = line.split(' ')
            line[9] = line[9].replace('nan', '0')
            line[9] = float(line[9])
            line[11] = float(line[11])
            temp_list.append(line)
    return temp_list


def to_percent(y, position):
    s = str(100 * round(y, 2))
    if plt.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def plotBar(x_pos, CTEs, error, label1, label2, title, limy, Nc, fname):
    # yerr = error.values() => for ax.bar function
    if len(CTEs) == 3:
        color_scheme = ['r', 'g', 'b']
    else:
        color_scheme = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs.values(), align='center', alpha=0.5, ecolor='black', color=color_scheme, capsize=10)
    ax.set_ylabel(label2)
    ax.set_xlabel(label1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(CTEs.keys())
    ax.set_title(title)
    #ax.yaxis.grid(True)
    #ax.set_axisbelow(True)
    #plt.tight_layout()
    plt.ylim(0, limy)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    # Consensus ratio rather than time
    if (ylabel.find('time') == -1):
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
    # plt.savefig(Nc + fname + '.png')
    plt.show()


def singlePlot(results, prmtrs, Nc, idx, vta, labelx, labely, pltt, limy, fname):
    consensu_dict = {}
    consensu_dict_std = {}
    for prmtr in prmtrs:
        consensu_dict[str(prmtr)] = []
        consensu_dict_std[str(prmtr)] = 0
    for result in results:
        if result[7] == Nc:
            consensu_dict[str(result[idx])].append(result[vta])
    for key in consensu_dict:
        consensu_dict_std[key] = np.std(consensu_dict[key])
        consensu_dict[key] = np.mean(consensu_dict[key])
    plotBar(prmtrs, consensu_dict, consensu_dict_std, labelx, labely, pltt, limy, Nc, fname)


def multipleCL(rslt, cl, Nc, prmtrs, net, idx, vta):
    for i in range(len(Nc)):
        if rslt[7] == Nc[i]:
            for j in range(len(prmtrs)):
                if rslt[idx] == prmtrs[j]:
                    cl[net][i][j].append(rslt[vta])


def multiplePlot(results, vars, prmtrs, Nc, idx, vta, labelx, labely, limy, fname):
    BA_dict = {}
    ERD_dict = {}
    ERS_dict = {}
    concensus_list = [[] * len(vars) for i in range(len(vars))]
    for i in range(len(concensus_list)):
        concensus_list[i] = [len(Nc) * [] for j in range(len(Nc))]
    for i in range(len(concensus_list)):
        for j in range(len(concensus_list[i])):
            concensus_list[i][j] = [len(prmtrs) * [] for k in range(len(prmtrs))]
    for result in results:
        if result[0] == vars[0]:
            network = 0
        elif result[0] == vars[1]:
            network = 1
        else:
            network = 2
        multipleCL(result, concensus_list, Nc, prmtrs, network, idx, vta)

    for i in range(len(vars)):
        if i == 0:
            for j in range(len(Nc)):
                for k in range(len(concensus_list[i][j])):
                    concensus_list[i][j][k] = np.mean(concensus_list[i][j][k])
                BA_dict[Nc[j]] = concensus_list[i][j]
        elif i == 1:
            for j in range(len(Nc)):
                for k in range(len(concensus_list[i][j])):
                    concensus_list[i][j][k] = np.mean(concensus_list[i][j][k])
                ERD_dict[Nc[j]] = concensus_list[i][j]
        else:
            for j in range(len(Nc)):
                for k in range(len(concensus_list[i][j])):
                    concensus_list[i][j][k] = np.mean(concensus_list[i][j][k])
                ERS_dict[Nc[j]] = concensus_list[i][j]

    X = np.arange(len(prmtrs))
    for i in range(len(Nc)):
        for j in range(len(prmtrs)):
            plt.bar(X[j] + 0.00, BA_dict[Nc[i]][j], color='r', width=0.25)
            plt.bar(X[j] + 0.25, ERD_dict[Nc[i]][j], color='g', width=0.25)
            plt.bar(X[j] + 0.50, ERS_dict[Nc[i]][j], color='b', width=0.25)
            plt.legend(variables)
            plt.xticks(X + 0.2, labelx)
            plt.ylabel(labely)
        plt.ylim(0, limy)
        plt.title(Nc[i] + ' consensus nodes')
        if (ylabel.find('time') == -1):
            formatter = FuncFormatter(to_percent)
            # Set the formatter
            plt.gca().yaxis.set_major_formatter(formatter)
        # plt.savefig(Nc[i] + fname + '.png')
        plt.show()


# ======== main =========== #

game_res = readConfigurations('TwoCirclesRes.txt')
title = ['10 consensus nodes', '20 coscensus nodes', '30 consensus nodes']
No_consensus = ['10', '20', '30']

"""""plotting adversaries graph for 10, 20, 30 consensus nodes"""
variables = None
parameters = ['0', '2', '5']
value_to_avg = -1
index = 3
xlabel = 'adversaries'
ylabel = 'consensus rate'
ylim = 1
figname = '_con_adv.png'
for i in range(len(No_consensus)):
    singlePlot(game_res, parameters, No_consensus[i], index, value_to_avg, xlabel, ylabel, title[i], ylim, figname)
#
# """"" plotting times of consensus graph for 10, 20, 30 consensus nodes"""
# variables = None
# parameters = ['0', '2', '5']
# value_to_avg = 9
# index = 3
# xlabel = 'adversaries'
# ylabel = 'time of consensus[sec]'
# ylim = 60
# figname = '_con_times'
# for i in range(len(No_consensus)):
#     singlePlot(game_res, parameters, No_consensus[i], index, value_to_avg, xlabel, ylabel, title[i], ylim, figname)
#
# """""plotting network graph for 10, 20, 30 consensus nodes"""
# variables = None
# parameters = ['barabasi-albert', 'erdos-renyi-dense', 'erdos-renyi-sparse']
# value_to_avg = -1
# index = 0
# xlabel = 'Network'
# ylabel = 'consensus rate'
# ylim = 1
# figname = '_con_net'
# for i in range(len(No_consensus)):
#     singlePlot(game_res, parameters, No_consensus[i], index, value_to_avg, xlabel, ylabel, title[i], ylim, figname)
#
"""""plotting visible's graph for 10, 20, 30 consensus nodes"""
variables = None
parameters = ['0', '1', '2', '5']
value_to_avg = -1
index = 5
xlabel = 'visibles'
ylabel = 'consensus rate'
ylim = 1
figname = '_con_vis'
for i in range(len(No_consensus)):
    singlePlot(game_res, parameters, No_consensus[i], index, value_to_avg, xlabel, ylabel, title[i], ylim, figname)
#
#
# """""plotting consensus graph for wide range visible's graph for 10, 20, 30 consensus nodes"""
# game_res = readConfigurations('WideRangeVisRes.txt')
# value_to_avg = -1
# index = 5
# xlabel = 'visibles'
# ylabel = 'consensus rate'
# ylim = 1
# figname = '_con_vis'
# variables = None
# for i in range(len(No_consensus)):
#     parameters = np.arange(int(No_consensus[i]) + 1)
#     singlePlot(game_res, parameters, No_consensus[i], index, value_to_avg, xlabel, ylabel, title[i], ylim, figname)
#
#
# """""plotting times graph for wide range visible's graph for 10, 20, 30 consensus nodes"""
# game_res = readConfigurations('WideRangeVisRes.txt')
# value_to_avg = 9
# index = 5
# xlabel = 'visibles'
# ylabel = 'time of consensus[sec]'
# ylim = 60
# figname = '_con_vis_times'
# variables = None
# for i in range(len(No_consensus)):
#     parameters = np.arange(int(No_consensus[i]) + 1)
#     singlePlot(game_res, parameters, No_consensus[i], index, value_to_avg, xlabel, ylabel, title[i], ylim, figname)

# """""plotting consensus rate for wide range adv's graph for 10, 20, 30 consensus nodes"""
# game_res = readConfigurations('WideRangeAdvRes.txt')
# value_to_avg = -1
# index = 3
# xlabel = 'adversaries'
# ylabel = 'consensus rate'
# ylim = 1
# figname = '_con_adv'
# variables = None
# for i in range(len(No_consensus)):
#     parameters = np.arange(16)
#     singlePlot(game_res, parameters, No_consensus[i], index, value_to_avg, xlabel, ylabel, title[i], ylim, figname)

# """""plotting times graph for wide range adv's graph for 10, 20, 30 consensus nodes"""
# # game_res = readConfigurations('WideRangeAdvRes.txt')
# value_to_avg = 9
# index = 3
# xlabel = 'adversaries'
# ylabel = 'time of consensus[sec]'
# ylim = 60
# figname = '_con_adv_times'
# variables = None
# for i in range(len(No_consensus)):
#     parameters = np.arange(16)
#     singlePlot(game_res, parameters, No_consensus[i], index, value_to_avg, xlabel, ylabel, title[i], ylim, figname)




# """""plotting consensus rate graph for network vs num of adversaries nodes for 10, 20, 30 consensus nodes"""
# variables = ['barabasi-albert', 'erdos-renyi-dense', 'erdos-renyi-sparse']
# parameters = ['0', '2', '5']
# value_to_avg = -1
# index = 3
# xticks = ['0 Adversaries', '2 Adversaries', '5 Adversaries']
# ylabel = 'consensus rate'
# ylim = 1
# figname = '_con_netVSadv'
# multiplePlot(game_results, variables, parameters, No_consensus, index, value_to_avg, xticks, ylabel, ylim, figname)
#
# """""plotting times graph for network vs num of adversaries nodes for 10, 20, 30 consensus nodes"""
# variables = ['barabasi-albert', 'erdos-renyi-dense', 'erdos-renyi-sparse']
# parameters = ['0', '2', '5']
# value_to_avg = 9
# index = 3
# xticks = ['0 Adversaries', '2 Adversaries', '5 Adversaries']
# ylabel = 'time of consensus [sec]'
# ylim = 60
# figname = '_times_netVSadv'
# multiplePlot(game_results, variables, parameters, No_consensus, index, value_to_avg, xticks, ylabel, ylim, figname)
#
# """""plotting consensus rate graph for network vs num of visible nodes for 10, 20, 30 consensus nodes"""
# variables = ['barabasi-albert', 'erdos-renyi-dense', 'erdos-renyi-sparse']
# parameters = ['0', '1', '2', '5']
# value_to_avg = -1
# index = 5
# xticks = ['0 visibles', '1 visibles', '2 visibles', '5 visibles']
# ylabel = 'consensus rate'
# ylim = 1
# figname = '_con_netVSvis'
# multiplePlot(game_results, variables, parameters, No_consensus, index, value_to_avg, xticks, ylabel, ylim, figname)

# """""plotting consensus rate graph for 1 visible with and with out strategy"""
# parameters = ['0', '2', '5']
# value_to_avg = -1
# index = 3
# xlabel = 'adversaries'
# ylabel = 'consensus rate'
# ylim = 1
# variables = ['two circles', 'full recognition']
# two_circles = singlePlot(two_circle_res, parameters, No_consensus, index, value_to_avg, xlabel, ylabel, plot_title, ylim, 'a')
# full_recognition = singlePlot(full_recognition_res, parameters, No_consensus, index, value_to_avg, xlabel, ylabel, plot_title, ylim, 'b')
#
# X = np.arange(len(parameters))
# for i in range(len(No_consensus)):
#     for j in range(len(parameters)):
#         plt.bar(X[j] + 0.00, two_circles[No_consensus[i]][j], color='r', width=0.25)
#         plt.bar(X[j] + 0.25, full_recognition[No_consensus[i]][j], color='g', width=0.25)
#         plt.legend(variables)
#         plt.xticks(X + 0.12, parameters)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#     plt.ylim(0, 1)
#     plt.title(No_consensus[i] + ' consensus nodes')
#     plt.show()

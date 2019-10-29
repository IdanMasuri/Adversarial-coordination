import numpy as np


# ====> generate general configuration here <====

def generateVis(num):
    return np.arange(num + 1)


graphnames = ['barabasi-albert', 'erdos-renyi-dense', 'erdos-renyi-sparse']
communicationtypes = ['unstructured']
batchtreatmenttype = ['adversarial']
numofadversaries = ['0', '2', '5']
visibilitymode = ['visibleconsensusnodes']
visibleconsensusnodes = ['1']
concensusnodes = ['concensnodes']
numofconcensusnodes = ['10', '20', '30']
strategy = ['nostrategy', 'onecircle', 'twocircles']


configurations = []
for gran in graphnames:
    for comt in communicationtypes:
        for batt in batchtreatmenttype:
            for numa in numofadversaries:
                for vism in visibilitymode:
                    for visc in visibleconsensusnodes:
                        for conc in concensusnodes:
                            for numc in numofconcensusnodes:
                                # for stra in strategy:
                                configurations.append([gran, comt, batt, numa, vism, visc, conc, numc])

with open('OneVisConfigs.txt', 'w') as configs:
    for row in configurations:
        row = [str(item) for item in row]
        configs.write(' '.join(row) + '\n')




import matplotlib.pyplot as plt
import numpy as np
import re

dataset = 'olympics'

decentration = np.loadtxt('../SpectralClustering_withRefine/result/{}/decentration.csv'.format(dataset))

nmi_list = []
with open('../SpectralClustering_withRefine/result/{}/log.txt'.format(dataset), 'r') as r:
    for line in r.readlines():
        if(re.match(r'.*NMI.*', line)):
            nmi = float(re.findall(r'.*NMI: ([\d|.]+).*', line)[0])
            nmi_list.append(nmi)

idx = [i for i in range(len(decentration))]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(idx, decentration, 'C0',label='decentration')

ax2 = ax1.twinx()
ax2.plot(idx, nmi_list,'C1',label='nmi')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='lower right')

ax1.set_xlabel('tri ({} dataset)'.format(dataset))
ax1.set_ylabel('decentration')
ax1.grid(True)
ax2.set_ylabel('nmi')

plt.savefig('../SpectralClustering_withRefine/result/{}/decentration_nmi.png'.format(dataset))
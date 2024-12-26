
# plot fig2: The performances of the reduced pilots-based beamforming
#            and full pilots versus SNR.
from utils import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# read data
untrain = [66.992, 81.312, 89.546, 93.916, 96.241, 97.418, 97.992]
train100 = [71.628, 84.509, 91.473, 95.083, 97.035, 97.992, 98.444]
train75 = [70.033, 82.625, 89.631, 93.265, 95.379, 96.532, 97.179]
train50 = [68.616, 81.362, 88.499, 92.129, 94.020, 95.083, 95.749]
train25 = [62.499, 74.046, 80.932, 84.868, 87.255, 88.898, 90.116]

# calculate practical data rate
untrain = np.array(untrain) * (1-0.0952)
train100 = np.array(train100)*(1-0.0952)
train75 = np.array(train75) * (1-0.0714)
train50 = np.array(train50) * (1-0.0476)
train25 = np.array(train25) * (1-0.0238)

fig,ax = plt.subplots(figsize=(10,8))
plt.xlabel('SNR (dB)', fontsize=24)
plt.ylabel('Ratio (%)', fontsize=24)
index = np.arange(len(untrain)) + 1
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xticks(index, (0,5,10,15,20,25,30))

plt.plot(index, train100, linewidth = 2, marker = '^', markersize=10,
         color = '#F65314', linestyle = '-', label=r'$\mathcal{\eta}=$'+'100% trained')
plt.plot(index, train75, linewidth = 2, marker = 's', markersize=10,
         color = '#FFBB00', linestyle = '-', label=r'$\mathcal{\eta}=$'+'75% trained')
plt.plot(index, train50, linewidth = 2, marker = 'o', markersize=10,
         color = '#7CBB00', linestyle = '-', label=r'$\mathcal{\eta}=$'+'50% trained')
plt.plot(index, train25, linewidth = 2, marker = 'd', markersize=10,
         color = '#00A1F1', linestyle = '-', label=r'$\mathcal{\eta}=$'+'25% trained')
plt.plot(index, untrain, linewidth = 2, marker = 'p', markersize=10,
         color = '#68217A', linestyle = '-', label=r'$\mathcal{\eta}=$'+'100% untrained')
plt.ylim([58, 94])
plt.legend(loc = 'lower right', fontsize=18)
ax.grid(True, ls=':', color='black', alpha=0.3)

# plot zoom figure
axins = inset_axes(ax, width='30%', height='31%', loc='lower left',
                   bbox_to_anchor=(0.34, 0.043, 0.5, 0.9),
                   bbox_transform=ax.transAxes)

# plot in subfigure
axins.plot(index, train100, linewidth = 2, marker = '^', markersize=10,
           color = '#F65314', linestyle = '-', label=r'$\mathcal{\eta}=$'+'100% trained')
axins.plot(index, train75, linewidth = 2, marker = 's', markersize=10,
           color = '#FFBB00', linestyle = '-', label=r'$\mathcal{\eta}=$'+'75% trained')
axins.plot(index, train50, linewidth = 2, marker = 'o', markersize=10,
           color = '#7CBB00', linestyle = '-', label=r'$\mathcal{\eta}=$'+'50% trained')
axins.plot(index, train25, linewidth = 2, marker = 'd', markersize=10,
           color = '#00A1F1', linestyle = '-', label=r'$\mathcal{\eta}=$'+'25% trained')
axins.plot(index, untrain, linewidth = 2, marker = 'p', markersize=10,
           color = '#68217A', linestyle = '-', label=r'$\mathcal{\eta}=$'+'100% untrained')

#subfigure set
axins.set_xlim(0.9, 1.1)
axins.set_ylim(64.7, 65.5)
axins.set_xticks([1])
axins.set_xticklabels([0], fontsize=18)
axins.set_yticks(np.arange(64.8, 65.5+0.1, 0.2))
axins.set_yticklabels([64.8,65.0,65.2,65.4], fontsize=18)
mark_inset(ax, axins, loc1=2, loc2=3, fc='none', ec='k', lw=1)
ax.grid(True, ls=':', color='black', alpha=0.3)


axins2 = inset_axes(ax, width='30%', height='30%', loc='lower left',
                   bbox_to_anchor=(0.795, 0.41, 0.5, 0.9),
                   bbox_transform=ax.transAxes)

# plot in subfigure
axins2.plot(index, untrain, linewidth = 2, marker = 'p', markersize=10,
            color = '#7030A0', linestyle = '-', label='100% pilot')
axins2.plot(index, train100, linewidth = 2, marker = '^', markersize=10,
            color = '#F25022', linestyle = '-', label='100% pilot with training')
axins2.plot(index, train50, linewidth = 2, marker = 'o', markersize=10,
           color = '#7FBA00', linestyle = '-', label='50% pilot with training')
axins2.plot(index, train25, linewidth = 2, marker = 'd', markersize=10,
            color = '#00A4EF', linestyle = '-', label='25% pilot with training')

#subfigure set
axins2.set_xlim(6.9, 7.1)
axins2.set_ylim(87.5, 89.5)
axins2.set_xticks([7])
axins2.set_xticklabels([30], fontsize=18)
axins2.set_yticks(np.arange(87.5, 89.5+0.1, 0.5))
axins2.set_yticklabels([87.5,88.0,88.5,89.0,89.5], fontsize=18)
mark_inset(ax, axins2, loc1=2, loc2=4, fc='none', ec='k', lw=1)
ax.grid(True, ls=':', color='black', alpha=0.3)

# save figure
plt.savefig('result/fig2.jpg')
plt.show()
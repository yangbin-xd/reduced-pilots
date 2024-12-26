
# plot fig6: The SEs of the proposed method in multi-UE scenarios versus SNR.
from utils import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# read data
SU2 = [3.647, 5.169, 6.613, 7.856, 8.715, 9.342, 9.701]
SU4 = [4.044, 5.623, 7.068, 8.395, 9.284, 9.938, 10.253]
SU8 = [4.234, 5.924, 7.539, 9.105, 10.185, 11.058, 11.366]
MU2 = [3.877, 5.757, 7.885, 9.862, 11.178, 11.922, 12.350]
MU4 = [4.112, 6.502, 9.359, 12.043, 13.790, 14.623, 15.055]
MU8 = [2.968, 5.410, 8.909, 12.512, 14.981, 15.987, 16.419]

# plot figure
fig,ax = plt.subplots(figsize=(10,8))
index = np.arange(len(SU2)) + 1
plt.xticks(index, (0,5,10,15,20,25,30))

plt.plot(index, SU2, linewidth = 2, marker = '^', markersize=10,
         color = '#F25022', linestyle = '-', label=r'$\mathbf{V}^{\mathrm{SU}}_{N=2}$')
plt.plot(index, MU2, linewidth = 2, marker = '^', markersize=10,
         color = '#F25022', linestyle = '--', label=r'$\mathbf{V}^{\mathrm{MU}}_{N=2}$')
plt.plot(index, SU4, linewidth = 2, marker = 's', markersize=10,
         color = '#FFB900', linestyle = '-', label=r'$\mathbf{V}^{\mathrm{SU}}_{N=4}$')
plt.plot(index, MU4, linewidth = 2, marker = 's', markersize=10,
         color = '#FFB900', linestyle = '--', label=r'$\mathbf{V}^{\mathrm{MU}}_{N=4}$')
plt.plot(index, SU8, linewidth = 2, marker = 'o', markersize=10,
         color = '#7FBA00', linestyle = '-', label=r'$\mathbf{V}^{\mathrm{SU}}_{N=8}$')
plt.plot(index, MU8, linewidth = 2, marker = 'o', markersize=10,
         color = '#7FBA00', linestyle = '--', label=r'$\mathbf{V}^{\mathrm{MU}}_{N=8}$')

# figure set
plt.ylim([2.2, 17])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('SNR (dB)', fontsize=24)
plt.ylabel('SE (bit/s/Hz)', fontsize=24)
plt.legend(loc = 'lower right', fontsize=18, ncol=3, framealpha=1,
           facecolor='white', handletextpad=0.5, columnspacing=0.5)
ax.grid(True, ls=':', color='black', alpha=0.3)

# plot in subfigure
axins = inset_axes(ax, width='30%', height='27%', loc='lower left',
                   bbox_to_anchor=(0.05, 0.38, 0.5, 1),
                   bbox_transform=ax.transAxes)

plt.plot(index, SU2, linewidth = 2, marker = '^', markersize=10,
         color = '#F25022', linestyle = '-', label='SU 2')
plt.plot(index, MU2, linewidth = 2, marker = '^', markersize=10,
         color = '#F25022', linestyle = '--', label='MU 2')
plt.plot(index, SU4, linewidth = 2, marker = 's', markersize=10,
         color = '#FFB900', linestyle = '-', label='SU 4')
plt.plot(index, MU4, linewidth = 2, marker = 's', markersize=10,
         color = '#FFB900', linestyle = '--', label='MU 4')
plt.plot(index, SU8, linewidth = 2, marker = 'o', markersize=10,
         color = '#7FBA00', linestyle = '-', label='SU 8')
plt.plot(index, MU8, linewidth = 2, marker = 'o', markersize=10,
         color = '#7FBA00', linestyle = '--', label='MU 8')

# subfigure set
axins.set_xlim(0.9, 1.1)
axins.set_ylim(2.7, 4.5)
axins.set_xticks([1])
axins.set_xticklabels([0], fontsize=18)
axins.set_yticks(np.arange(3.0, 4.5+0.5, 0.5))
axins.set_yticklabels([3.0,3.5,4.0,4.5], fontsize=18)
mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='k', lw=1)
ax.grid(True, ls=':', color='black', alpha=0.3)

# plot in sub-subfigure
axins2 = inset_axes(ax, width='30%', height='27%', loc='lower left',
                    bbox_to_anchor=(0.265, 0.65, 0.5, 1),
                    bbox_transform=ax.transAxes)

plt.plot(index, SU2, linewidth = 2, marker = '^', markersize=10,
         color = '#F65314', linestyle = '-', label='SU 2')
plt.plot(index, MU2, linewidth = 2, marker = '^', markersize=10,
         color = '#F65314', linestyle = '--', label='MU 2')
plt.plot(index, SU4, linewidth = 2, marker = 's', markersize=10,
         color = '#FFBB00', linestyle = '-', label='SU 4')
plt.plot(index, MU4, linewidth = 2, marker = 's', markersize=10,
         color = '#FFBB00', linestyle = '--', label='MU 4')
plt.plot(index, SU8, linewidth = 2, marker = 'o', markersize=10,
         color = '#7CBB00', linestyle = '-', label='SU 8')
plt.plot(index, MU8, linewidth = 2, marker = 'o', markersize=10,
         color = '#7CBB00', linestyle = '--', label='MU 8')

# sub-subfigure set
axins2.set_xlim(0.98, 1.02)
axins2.set_ylim(3.97, 4.3)
axins2.set_xticks([1])
axins2.set_xticklabels([0], fontsize=18)
axins2.set_yticks(np.arange(4.0, 4.3+0.1, 0.1))
axins2.set_yticklabels([4.0,4.1,4.2,4.3], fontsize=18)
mark_inset(axins, axins2, loc1=2, loc2=4, fc="none", ec='k', lw=1)
ax.grid(True, ls=':', color='black', alpha=0.3)

# save figure
plt.savefig('result/fig10.jpg')
plt.show()
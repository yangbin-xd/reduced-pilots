
# plot fig5: The performances of GBM, CKM, and radio map methods
#            under LoS and NLoS conditions vs. positioning error.
from utils import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# read data
GBM_LoS = [91.162, 90.488, 89.066, 86.646, 84.411, 81.703]
GBM_NLoS = [21.589, 21.414, 21.039, 20.938, 21.243, 21.633]
CKM_LoS = [86.087, 86.398, 84.385, 84.067, 81.931, 79.360]
CKM_NLoS = [75.044, 73.850, 68.401, 65.793, 62.766, 56.123]
RM_LoS = [88.584, 86.970, 84.302, 82.168, 79.326, 76.893]
RM_NLoS = [82.090, 81.392, 79.567, 77.168, 74.737, 72.214]

# plot figure1
fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,8), 
                              gridspec_kw={'height_ratios': [5, 1]})
index = np.arange(6)
plt.xticks(index, (0,1,2,3,4,5))
ax.plot(index, GBM_LoS, linewidth = 2, marker = '^', markersize=10,
        color = '#F65314', linestyle = '-', label='GBM LoS')
ax.plot(index, CKM_LoS, linewidth = 2, marker = 's', markersize=10,
        color = '#FFBB00', linestyle = '-', label='CKM LoS')
ax.plot(index, RM_LoS, linewidth = 2, marker = 'o', markersize=10,
        color = '#7CBB00', linestyle = '-', label='Radio map LoS')
# ax.plot(index, GBM_NLoS, linewidth = 2, marker = '^', markersize=10,
#          color = '#F65314', linestyle = '--', label='GBM NLoS')
ax.plot(index, CKM_NLoS, linewidth = 2, marker = 's', markersize=10,
        color = '#FFBB00', linestyle = '--', label='CKM NLoS')
ax.plot(index, RM_NLoS, linewidth = 2, marker = 'o', markersize=10,
        color = '#7CBB00', linestyle = '--', label='Radio map NLoS')
for label in ax.get_yticklabels():
    label.set_fontsize(24)
ax.set_ylim(54, 95)
ax.tick_params(bottom=False)

# plot figure2
ax2.plot(index, GBM_LoS, linewidth = 2, marker = '^', markersize=10,
        color = '#F65314', linestyle = '-', label='GBM LoS')
ax2.plot(index, CKM_LoS, linewidth = 2, marker = 's', markersize=10,
        color = '#FFBB00', linestyle = '-', label='CKM LoS')
ax2.plot(index, RM_LoS, linewidth = 2, marker = 'o', markersize=10,
        color = '#7CBB00', linestyle = '-', label='Radio map LoS')
ax2.plot(index, GBM_NLoS, linewidth = 2, marker = '^', markersize=10,
         color = '#F65314', linestyle = '--', label='GBM NLoS')
ax2.plot(index, CKM_NLoS, linewidth = 2, marker = 's', markersize=10,
        color = '#FFBB00', linestyle = '--', label='CKM NLoS')
ax2.plot(index, RM_NLoS, linewidth = 2, marker = 'o', markersize=10,
        color = '#7CBB00', linestyle = '--', label='Radio map NLoS')
ax2.set_ylim(18, 26)
ax2.set_ylabel('', fontsize=24)
for label2 in ax2.get_xticklabels():
    label2.set_fontsize(24)
for label2 in ax2.get_yticklabels():
    label2.set_fontsize(24)

# figure set
fig.text(0.04, 0.5, 'Ratio (%)', va='center', rotation='vertical', fontsize=24)
fig.subplots_adjust(hspace=0.05)
y_major_locator = MultipleLocator(5)
ax2.yaxis.set_major_locator(y_major_locator)
ax2.legend(fontsize=18, loc=(0.03,0.66), framealpha=1, facecolor='white')
ax.grid(True, ls=':', color='black', alpha=0.3)
ax2.grid(True, ls=':', color='black', alpha=0.3)

# plot break mark
d = 0.01
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, lw=1)
ax.plot((-d, +d), (-d/5, +d/5), **kwargs)
ax.plot((1 - d, 1 + d), (-d/5, +d/5), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

# figure set
plt.xlabel(r'$\sigma_e$ (m)', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
ax.grid(True, ls=':', color='black', alpha=0.3)

# save figure
plt.savefig('result/fig4.jpg')
plt.show()
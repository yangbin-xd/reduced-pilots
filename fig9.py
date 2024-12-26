
# plot fig10: The performances of integration scheme without and
#             with SVM-based discriminator versus positioning error.
from utils import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# read data
RP = [87.744, 87.744, 87.744, 87.744, 87.744, 87.744]
RM = [85.932, 84.692, 82.369, 80.126, 77.452, 74.982]
I = [89.318, 89.269, 89.198, 89.102, 89.004, 88.946]
svm = [90.447, 90.060, 89.578, 89.322, 89.081, 89.018]
best = [91.178, 91.041, 90.733, 90.418, 90.179, 89.961]

# plot figure
fig,ax = plt.subplots(figsize=(10,8))
index = np.arange(len(RM)) + 1
plt.xticks(index, (0,1,2,3,4,5))

plt.plot(index, RP, linewidth = 2, marker = '^', markersize=10,
         color = '#F25022', linestyle = '-', label='Reduced pilots with '+r'$\eta=50\%$')
plt.plot(index, RM, linewidth = 2, marker = 'o', markersize=10,
         color = '#FFB900', linestyle = '-', label='Radio map')
plt.plot(index, I, linewidth = 2, marker = 's', markersize=10,
         color = '#7FBA00', linestyle = '-', label='Integration')
plt.plot(index, svm, linewidth = 2, marker = 'd', markersize=10,
         color = '#00A1F1', linestyle = '-', label='Integration with SVM-based discriminator')
plt.plot(index, best, linewidth = 2, marker = 'p', markersize=10,
         color = '#68217A', linestyle = '-', label='Integration with error-free discriminator')

# figure set
plt.xlabel(r'$\sigma_e$ (m)', fontsize=24)
plt.ylabel('Ratio (%)', fontsize=24)
plt.ylim([74, 92])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(loc = 'lower left', fontsize=18)
ax.grid(True, ls=':', color='black', alpha=0.3)

# plot in subfigure
axins2 = inset_axes(ax, width='36.0%', height='33.5%', loc='lower left',
                   bbox_to_anchor=(0.764, 0.322, 0.5, 1),
                   bbox_transform=ax.transAxes)

axins2.plot(index, RP, linewidth = 2, marker = '^', markersize=10,
            color = '#F25022', linestyle = '-', label='Reduced pilots with '+r'$\eta=50\%$')
axins2.plot(index, I, linewidth = 2, marker = '^', markersize=10,
            color = '#7FBA00', linestyle = '-', label='Integration')
axins2.plot(index, svm, linewidth = 2, marker = 's', markersize=10,
            color = '#00A1F1', linestyle = '-', label='Integration with SVM-based discriminator')
axins2.plot(index, best, linewidth = 2, marker = 'p', markersize=10,
            color = '#68217A', linestyle = '-', label='Integration with error-free discriminator')

# subfigure set
axins2.set_xlim(5.9, 6.1)
axins2.set_ylim(88.90, 89.05)
axins2.set_xticks([6])
axins2.set_xticklabels([5], fontsize=18)
axins2.set_yticks(np.arange(88.90, 89.05+0.01, 0.05))
axins2.set_yticklabels([88.90,88.95,89.0,89.05], fontsize=18)
axins2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
mark_inset(ax, axins2, loc1=2, loc2=4, fc='none', ec='k', lw=1)
ax.grid(True, ls=':', color='black', alpha=0.3)

# save figure
# plt.savefig('result/fig101.pdf')
plt.show()
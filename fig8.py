
# plot fig10: The performances of integration scheme without and
#             with SVM-based discriminator versus SNR.
from utils import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# read data
RP = [65.350, 77.489, 84.286, 87.744, 89.544, 90.557, 91.191]
RM = [75.436, 79.340, 82.338, 84.692, 86.580, 88.125, 89.408]
I = [71.761, 81.015, 85.984, 89.269, 90.740, 91.543, 92.049]
svm = [79.348, 83.658, 87.561, 90.060, 91.557, 92.500, 92.893]
best = [81.115, 85.259, 88.608, 91.040, 92.239, 93.068, 93.653]

# plot figure
fig,ax = plt.subplots(figsize=(10,8))
index = np.arange(len(RM)) + 1
plt.xticks(index, (0,5,10,15,20,25,30))

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
plt.xlabel('SNR (dB)', fontsize=24)
plt.ylabel('Ratio (%)', fontsize=24)
plt.ylim([63, 95])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(loc = 'lower right', fontsize=18)
ax.grid(True, ls=':', color='black', alpha=0.3)

# plot in subfigure
axins2 = inset_axes(ax, width='30%', height='31%', loc='lower left',
                   bbox_to_anchor=(0.795, 0.364, 0.5, 1),
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
axins2.set_xlim(6.9, 7.1)
axins2.set_ylim(91.7, 94.0)
axins2.set_xticks([7])
axins2.set_xticklabels([30], fontsize=18)
axins2.set_yticks(np.arange(92.0, 94.0+0.5, 0.5))
axins2.set_yticklabels([92.0,92.5,93.0,93.5,94.0], fontsize=18)
mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec='k', lw=1)
ax.grid(True, ls=':', color='black', alpha=0.3)

# save figure
plt.savefig('result/fig10.pdf')
plt.show()
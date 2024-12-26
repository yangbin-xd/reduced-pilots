
# plot fig1: The training and validation losses of
#            reduced pilotsbased beamforming versus epochs.
from utils import *
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# plot training loss and validation loss vs. epoch
fig,ax = plt.subplots(figsize=(10,8))
eta = np.array([100, 75, 50, 25])
color = ["#F65314", "#FFBB00", "#7CBB00", "#00A1F1"]

for i in range(eta.shape[0]):
    with open(f'loss/RP{FR}_{eta[i]}.json', 'r') as f:
        loss_dict = json.load(f)
    ax.plot(loss_dict['loss'], c=color[i], linestyle = '-', linewidth=2.5,
            label=r'$\eta=$'+f'{eta[i]}% training loss')
    if 'val_loss' in loss_dict:
        ax.plot(loss_dict['val_loss'], c=color[i], linestyle = '--', linewidth=2.5,
                label=r'$\eta=$'+f'{eta[i]}% validation loss')

# figure set
plt.xlabel('Epochs', fontsize=26)
plt.ylabel('Loss (bit/s/Hz)', fontsize=26)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.ylim([-13.7, -9.6])
plt.legend(loc='upper right', fontsize=20, ncol=2, framealpha=1, facecolor='white')
ax.grid(True, ls=':', color='black', alpha=0.3)
plt.tight_layout()

# save figure
plt.savefig('result/fig1.pdf')
plt.show()
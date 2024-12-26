
# plot fig7: The training and validation losses of integration scheme versus epochs.
from utils import *
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# plot training loss and validation loss vs. epoch
fig,ax = plt.subplots(figsize=(10,8))
SNR = np.arange(0,30+1,5)
color = ['#F65314', '#FFBB00', '#FFEC40', '#7CBB00', '#00CCCC', '#00A1F1', '#68217A']

for i in range(SNR.shape[0]):
    with open(f'loss/I{FR}_{SNR[i]}.json', 'r') as f:
        loss_dict = json.load(f)
    ax.plot(loss_dict['loss'], c=color[i], linestyle = '-', linewidth=2,
            label=f'{SNR[i]}dB training loss')
    if 'val_loss' in loss_dict:
        ax.plot(loss_dict['val_loss'], c=color[i], linestyle = '--', linewidth=2,
                label=f'{SNR[i]}dB validation loss')

# figure set
plt.xlabel('Epochs', fontsize=24)
plt.ylabel('Loss (bit/s/Hz)', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xticks(range(0, 210, 50))
plt.ylim([-14.0, -2])
plt.legend(loc='upper right', fontsize=18, ncol=1, bbox_to_anchor=(1.7, 1))
plt.subplots_adjust(right=0.6)
ax.grid(True, ls=':', color='black', alpha=0.3)

# save figure
plt.savefig('result/fig5.jpg')
plt.show()
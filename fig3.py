
# plot fig3: The training and validation losses of
#            radio map-based beamforming vs. epochs.
from utils import *
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# read data
with open(f'loss/RM{FR}.json', 'r') as f:
    history_dict = json.load(f)

# plot training loss and validation loss vs. epoch
fig,ax = plt.subplots(figsize=(10,8.0))
ax.plot(history_dict['loss'], label='Training loss', linewidth=2.5, c='#F65314')
if 'val_loss' in history_dict:
    ax.plot(history_dict['val_loss'], label='Validation loss', linewidth=2.5, c='#7CBB00')

# figure set
plt.xlabel('Epochs', fontsize=28)
plt.ylabel('Loss (bit/s/Hz)', fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.ylim([-13.2, -9.0])
plt.legend(fontsize=24)
plt.tight_layout()
ax.grid(True, ls=':', color='black', alpha=0.3)

# save figure
plt.savefig('result/fig3.pdf')
plt.show()
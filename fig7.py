
# plot fig9: The classification results of SVM-based discriminator
#            at different SNRs.
from utils import *
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import MultipleLocator
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# import data
from label import x_train, x_test, y_train, y_test

# set parameters
SNR = 30

# local SNR or global SNR
if __name__ == '__main__':
    SNR = SNR
    np.save('data/SNR.npy', SNR)
else:
    SNR = np.load('data/SNR.npy')

# load model
with open('model/svm1_1_{}.pkl'.format(SNR), 'rb') as file:
    best_model = pickle.load(file)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFBB00', '#7CBB00'])

    fig, ax = plt.subplots(figsize=(10, 3))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)
    
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='#FFBB00',
                label='Radio map better', edgecolors='k')

    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='#7CBB00',
                label='Integration better', edgecolors='k')
    
    ax.invert_xaxis()
    plt.legend(loc='upper right', fontsize=18, handletextpad=0.2, handlelength=1)
    plt.xlabel('X (m)', fontsize=18)
    plt.ylabel('Y (m)', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    x_major_locator=MultipleLocator(25)
    y_major_locator=MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.tight_layout()

# plot figure
plot_decision_boundary(best_model, x_train, y_train)
plt.savefig(f'result/fig7_{SNR}.jpg')
# plot_decision_boundary(best_model, x_test, y_test)
# plt.savefig(f'result/test{SNR}.jpg')
plt.show()
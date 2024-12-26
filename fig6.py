
# plot fig8: The heatmap of SVM classification accuracies with
#            different combinations of C and γ at 0dB.
from utils import *
import pickle
import seaborn as sns
from sklearn.svm import SVC
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# set parameters
loc_std = 1
fading_ratio = 0.5
SNR = 15

# local parameters or global parameters
if __name__ == '__main__':
    loc_std = loc_std
    fading_ratio = fading_ratio
    SNR = SNR
    np.save('data/loc_std.npy', loc_std)
    np.save('data/fading_ratio.npy', fading_ratio)
    np.save('data/SNR.npy', SNR)
else:
    loc_std = np.load('data/loc_std.npy')
    fading_ratio = np.load('data/fading_ratio.npy')
    SNR = np.load('data/SNR.npy')

    # import data
from label import x_train, x_test, y_train, y_test

# search for best parameters
param_grid = {'C': [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
              'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
              'kernel': ['rbf']}

# save time
param_grid = {'C': [1, 10, 100],
              'gamma': [0.001, 0.01, 0.1],
              'kernel': ['rbf']}

# svm
svm_classifier = SVC()
grid_search = GridSearchCV(svm_classifier, param_grid, refit=True, verbose=0, cv=5)
grid_search.fit(x_train, y_train)
scores = grid_search.cv_results_['mean_test_score']
scores_matrix = (scores.reshape(len(param_grid['C']), len(param_grid['gamma'])) * 100)
best_model = grid_search.best_estimator_
with open('model/svm{}_{}_{}_{}.pkl'.format(FR, loc_std, fading_ratio, SNR), 'wb') as file:
    pickle.dump(best_model, file)

# print results
print("best parameters:", grid_search.best_params_)
print("training loss:", np.round(grid_search.best_score_*100, 3), '%')
test_score = best_model.score(x_test, y_test)
print('Accuracy of test set:', np.round(test_score*100,3), '%')

# plot score matrix
plt.figure(figsize=(10, 10))
ax = plt.gca()
sns.heatmap(scores_matrix, annot=False, fmt='', xticklabels=param_grid['gamma'], 
            yticklabels=param_grid['C'], cmap='viridis', ax=ax, cbar=False)
for i in range(scores_matrix.shape[0]):
    for j in range(scores_matrix.shape[1]):
        text = ax.text(j+0.5, i+0.5, f"{scores_matrix[i, j]:.1f}%",
                       ha="center", va="center", color="white", fontsize=24)

# figure set
plt.xlabel(r'$\gamma$', fontsize=30)
plt.ylabel(r'$C$', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# custom label
def custom_label(x):
    if int(np.log10(x)) == 0:
        return "1"
    else:
        return r"$10^{{{}}}$".format(int(np.log10(x)))

xticklabels = [custom_label(x) for x in param_grid['gamma']]
yticklabels = [custom_label(y) for y in param_grid['C']]
ax.set_xticklabels(xticklabels, fontsize=30)
ax.set_yticklabels(yticklabels, fontsize=30, rotation=0)
plt.tight_layout()

# # save figure
# plt.savefig('result/fig8.pdf')
# plt.show()
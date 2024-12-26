
# integration scheme with svm-based discriminator
from utils import *
import pickle

# set parameters
loc_std = 1
fading_ratio = 0.1
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
from reduced_pilot import opt_SE, RP2opt
from radio_map import RM2opt, V_RM_test
from integrate import I2opt, V_I_test
from label import x_test, RM_SE_test, I_SE_test, best_SE_test

# load model
with open('model/svm{}_{}_{}.pkl'.format(FR, loc_std, SNR), 'rb') as file:
    best_model = pickle.load(file)

# test
y_pred = best_model.predict(x_test)
N_test = RM_SE_test.shape[0]
svm_SE_test = np.empty(N_test)
V_svm_test = np.empty(V_RM_test.shape, dtype='complex')
for i in range(N_test):
    if y_pred[i] == 0:
        svm_SE_test[i] = RM_SE_test[i]
        V_svm_test[i,:] = V_RM_test[i,:]
    else:
        svm_SE_test[i] = I_SE_test[i]
        V_svm_test[i,:] = V_I_test[i,:]

# calculate ratio
RP2opt = RP2opt * (1-0.0476)
I2opt = I2opt * (1-0.0476)
best2opt = np.mean(best_SE_test) / np.mean(opt_SE) * 100
svm2opt = np.mean(svm_SE_test) / np.mean(opt_SE) * 100

# print result
if __name__ == '__main__':
    print('RP2opt:', np.round(RP2opt, 3), '%')
    print('RM2opt:', np.round(RM2opt, 3), '%')
    print('I2opt:', np.round(I2opt, 3), '%')
    print('svm2opt:', np.round(svm2opt, 3), '%')
    print('best2opt:', np.round(best2opt, 3), '%')
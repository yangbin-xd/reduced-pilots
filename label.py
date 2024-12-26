
# svm-based discriminator
from utils import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# import beamforming vectors of radio map and integration
from radio_map import V_RM_train, V_RM_test, CSI_train, CSI_test
from integrate import CSI_fading, CSI_noise, V_I_train, V_I_test

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


# process data
[UEloc, CSI] = eliminate_block(UEloc, CSI)
UEloc[:,[0,1]] = UEloc[:,[1,0]]
x_train, x_test = train_test_split(UEloc, test_size=0.2, random_state=1)

# noise
power = (np.linalg.norm(CSI_fading))**2/np.prod(CSI_fading.shape)
noise = power / (10**(SNR/10))

# calculate SE of radio map and integration scheme
RM_SE_train = cal_SE(CSI_train, V_RM_train, noise)
RM_SE_test = cal_SE(CSI_test, V_RM_test, noise)
I_SE_train = cal_SE(CSI_train, V_I_train, noise) * (1-0.0476)
I_SE_test = cal_SE(CSI_test, V_I_test, noise) * (1-0.0476)

# the ratio of train set and validation set
ratio = 0.5
N_train = RM_SE_train.shape[0]
N_test = RM_SE_test.shape[0]
x_train = x_train[int(N_train*ratio):,:]
RM_SE_train = RM_SE_train[int(N_train*ratio):]
I_SE_train = I_SE_train[int(N_train*ratio):]

# add binary label for classification
def add_label(RM_SE, I_SE):

    '''
    input: RM_SE, I_SE (N,)
    output: RM_point, I_point, svm_SE, label (N,)
    
    '''
    RM_point, I_point = [], []
    label = np.zeros(RM_SE.shape[0])
    best_SE = np.empty(RM_SE.shape)
    for i in range(RM_SE.shape[0]):
        if RM_SE[i] > I_SE[i]:
            RM_point.append(i)
            best_SE[i] = RM_SE[i]
        else:
            I_point.append(i)
            best_SE[i] = I_SE[i]
            label[i] = 1
    return RM_point, I_point, best_SE, label

# add binary label
RM_point_train, I_point_train, best_SE_train, y_train = add_label(RM_SE_train, I_SE_train)
RM_point_test, I_point_test, best_SE_test, y_test = add_label(RM_SE_test, I_SE_test)
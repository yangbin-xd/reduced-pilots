
# radio map
from utils import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

# set parameters
loc_std = 0
fading_ratio = 0.1
SNR = 15

# local parameters or global parameters
if __name__ == '__main__':
    loc_std = loc_std
    fading_ratio = fading_ratio
    SNR = SNR
else:
    SNR = np.load('data/SNR.npy')
    fading_ratio = np.load('data/fading_ratio.npy')
    loc_std = np.load('data/loc_std.npy')

# process data
[UEloc, CSI] = eliminate_block(UEloc, CSI)
UEloc_error = add_error(UEloc, loc_std)
loc_norm = normalize_loc(UEloc_error)
CSI_fading = add_fading(CSI, fading_ratio)
CSI_noise = add_noise(CSI_fading, SNR).astype(np.complex64)

# divide train and test set
x_train, x_test, y_train, y_test = train_test_split(loc_norm, CSI_noise, test_size=0.2, random_state=1)
CSI_train, CSI_test = train_test_split(CSI_fading, test_size=0.2, random_state=1)

# noise
# CSI_fading = CSI
power = (np.linalg.norm(CSI_fading))**2/np.prod(CSI_fading.shape)
noise = power / (10**(SNR/10))

# model of radio map
def radio_map_model():

    '''
    input: normalized location (2,)
    output: complete beamforming vector (Nc, Nt, 1)

    '''
    input = layers.Input(shape=(2,))
    x1 = layers.Dense(16, activation='relu')(input)
    x2 = layers.Dense(128, activation='relu')(x1)
    x3 = layers.Dense(Nc * Nt * 2, activation='tanh')(x2)
    x4 = layers.Reshape([Nc, Nt, 2])(x3)
    output = tf.complex(x4[:,:,:,0], x4[:,:,:,1])[:,:,:,None]
    return models.Model(input, output)

# # keep to train while annotate to test
# model, history = train(x_train, y_train, x_test, y_test, radio_map_model(), noise, 1000)
# model.save(f'model/RM{FR}.h5')
# for key in history.history:
#     history.history[key] = [float(i) for i in history.history[key]]
# with open(f'loss/RM{FR}.json', 'w') as f:
#     json.dump(history.history, f)

# test
model = models.load_model(f'model/RM{FR}.h5',
                          custom_objects={'cust_loss': su_loss(noise)})
V_RM_train = model.predict(x_train)
V_RM_test = model.predict(x_test)

# normalize beamforming vector
V_RM_train = normalize_V(V_RM_train)
V_RM_test = normalize_V(V_RM_test)

# calculate SE of radio map
RM_SE = cal_SE(CSI_test, V_RM_test, noise)

# calculate optimum
U, Sigma, VT = np.linalg.svd(CSI_test)
V_opt = VT[:,:,0,:][:,:,:,None].conjugate()
opt_SE = cal_SE(CSI_test, V_opt, noise)

# calculate ratio
RM2opt = np.mean(RM_SE) / np.mean(opt_SE) * 100

# calculate LoS and NLoS conditions
RM_LoS_SE, RM_NLoS_SE = compute(RM_SE)
opt_LoS_SE, opt_NLoS_SE = compute(opt_SE)
RM2opt_LoS = np.mean(RM_LoS_SE) / np.mean(opt_LoS_SE) * 100
RM2opt_NLoS = np.mean(RM_NLoS_SE) / np.mean(opt_NLoS_SE) * 100

# print result
if __name__ == '__main__':
    print('RM2opt:', np.round(RM2opt, 3), '%')
    print('RM LoS:', np.round(RM2opt_LoS, 3), '%')
    print('RM NLoS:', np.round(RM2opt_NLoS, 3), '%')

# Integrate radio map and reduced pilot
from utils import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

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

# import beamforming vectors of reduced pilot and radio map
from reduced_pilot import CSI_fading, CSI_noise, V_RP_train, V_RP_test
from radio_map import V_RM_train, V_RM_test, CSI_train, CSI_test

# divide train and test set
x_train = np.concatenate([np.real(V_RP_train), np.imag(V_RP_train),
                          np.real(V_RM_train), np.imag(V_RM_train)], axis=-1)
x_test = np.concatenate([np.real(V_RP_test), np.imag(V_RP_test),
                         np.real(V_RM_test), np.imag(V_RM_test)], axis=-1)
y_train, y_test = train_test_split(CSI_noise, test_size=0.2, random_state=1)

# noise
power = (np.linalg.norm(CSI_fading))**2/np.prod(CSI_fading.shape)
noise = power / (10**(SNR/10))

# model of integration
def integration_model():

    '''
    input: concatenated beamforming vector (Nc, Nt, 4)
    output: integrated beamforming vector (Nc, Nt, 1)

    '''
    input = layers.Input(shape=(Nc, Nt, 4))
    x1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(input)
    x2 = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x1)
    x3 = layers.Conv2D(2, (3,3), padding='same', activation='tanh')(x2)
    output = tf.complex(x3[:,:,:,0], x3[:,:,:,1])[:,:,:,None]
    return models.Model(input, output)

# keep to train while annotate to test
if __name__ == '__main__':
    model, history = train(x_train, y_train, x_test, y_test, integration_model(), noise, 200)
    model.save(f'model/I{FR}_{loc_std}_{SNR}.h5')
    # for key in history.history:
    #     history.history[key] = [float(i) for i in history.history[key]]
    # with open(f'loss/I{FR}_{SNR}.json', 'w') as f:
    #     json.dump(history.history, f)

# test
model = models.load_model(f'model/I{FR}_{loc_std}_{SNR}.h5',
                          custom_objects={'cust_loss': su_loss(noise)})
V_I_train = model.predict(x_train)
V_I_test = model.predict(x_test)

# normalize beamforming vector
V_I_train = normalize_V(V_I_train)
V_I_test = normalize_V(V_I_test)

# calculate SE of integration scheme
I_SE = cal_SE(CSI_test, V_I_test, noise)

# calculate optimum
U, Sigma, VT = np.linalg.svd(CSI_test)
V_opt = VT[:,:,0,:][:,:,:,None].conjugate()
opt_SE = cal_SE(CSI_test, V_opt, noise)

# calculate ratio
I2opt = np.mean(I_SE) / np.mean(opt_SE) * 100

# print result
if __name__ == '__main__':
    print('I2opt:', np.round(I2opt, 3), '%')
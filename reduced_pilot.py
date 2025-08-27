
# reduced pilots
from utils import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

# set parameters
fading_ratio = 0.1
SNR = 30
eta = 50 # 100, 75, 50, 25

# local parameters or global parameters
if __name__ == '__main__':
    fading_ratio = fading_ratio
    SNR = SNR
else:
    fading_ratio = np.load('data/fading_ratio.npy')
    SNR = np.load('data/SNR.npy')

# process data
[UEloc, CSI] = eliminate_block(UEloc, CSI)
CSI_fading = add_fading(CSI, fading_ratio)
CSI_RB = np.repeat(CSI_fading, 14, axis=2) # (N, Nc, Ns, Nt)
CSI_noise = add_noise(CSI_RB, SNR) # (N, Nc, Ns, Nt)

# pilot measurement
Nc_pos = [2,3,4,5,8,9,10,11]
CSI_pilot = np.zeros([CSI_noise.shape[0], Nt], dtype='complex')
for i in range(CSI_noise.shape[0]):
    for j in range(len(Nc_pos)):
        for k in range(int(Nt/len(Nc_pos))):
            CSI_pilot[i, 4*j+k] = CSI_noise[i, Nc_pos[j], k+3, 4*j+k]

# estimate CSI by reduced pilots
def estimate_CSI(eta):
    if eta == 100:
        return
    elif eta == 75:
        for n in np.arange(3,Nt,4):
            CSI_pilot[:,n] = 0
    elif eta == 50:
        for n in np.arange(1,Nt,2):
            CSI_pilot[:,n] = 0
    elif eta == 25:
        for n in np.arange(1,Nt,4):
            for j in range(3):
                CSI_pilot[:,n+j] = 0
    else:
        raise ValueError('Please choose from 100,75,50,25.')
    
estimate_CSI(eta)
CSI_part = CSI_pilot[:,None,None,:].repeat(Nc, axis=1)
U, Sigma, VT = np.linalg.svd(CSI_part)
reduced_pilot = VT[:,:,0,:] # (N, Nc, Nt)
pilot_comp = np.concatenate([np.real(reduced_pilot[:,:,:,None]),
             np.imag(reduced_pilot[:,:,:,None])], axis=-1) # (N, Nc, Nt, 2)

# divide train and test set
CSI_noise = add_noise(CSI_fading, SNR).astype(np.complex64)
x_train, x_test, y_train, y_test = train_test_split(pilot_comp, CSI_noise, test_size=0.2, random_state=1)
CSI_train, CSI_test = train_test_split(CSI_fading, test_size=0.2, random_state=1)
N_train, N_test = x_train.shape[0], x_test.shape[0]

# noise
power = (np.linalg.norm(CSI_fading))**2/np.prod(CSI_fading.shape)
noise = power / (10**(SNR/10))

# model of reduced pilot
def reduced_pilot_model():

    '''
    input: partial CSI estimation (Nc, Nt, 2) with zero padding
    output: complete beamforming vector (Nc, Nt, 1)

    '''
    def ResNet(input):
        x1 = layers.Conv2D(16, (3,3), padding='same', activation='relu')(input)
        x2 = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x1)
        x3 = layers.Conv2D(2, (3,3), padding='same', activation='tanh')(x2)
        return layers.add([input, x3])
    input = layers.Input(shape=(Nc, Nt, 2))
    x = input
    for _ in range(1):
        x = ResNet(x)
    x = layers.Conv2D(2, (3,3), padding='same', activation='tanh')(x)
    output = tf.complex(x[:,:,:,0], x[:,:,:,1])[:,:,:,None]
    return models.Model(input, output)

# # keep to train while annotate to test
# model, history = train(x_train, y_train, x_test, y_test, reduced_pilot_model(), noise, 100)
# model.save(f'model/RP{FR}_{eta}.h5')
# for key in history.history:
#     history.history[key] = [float(i) for i in history.history[key]]
# with open(f'loss/RP{FR}_{eta}.json', 'w') as f:
#     json.dump(history.history, f)

# test
# trained
model = models.load_model(f'model/RP{FR}_{eta}.h5', custom_objects={'cust_loss': su_loss(noise)})
V_RP_train = model.predict(x_train)
V_RP_test = model.predict(x_test)

# # keep only for eta = 100% untrained 
# x_test_comp = x_test[:,:,:,0] + 1j * x_test[:,:,:,1]
# V_RP_test = x_test_comp.conjugate()[:,:,:,None]

# normalize beamforming vector
V_RP = normalize_V(V_RP_test)

# calculate SE of GBM
RP_SE = cal_SE(CSI_test, V_RP, noise)

# calculate optimum
U, Sigma, VT = np.linalg.svd(CSI_test)
V_opt = VT[:,:,0,:][:,:,:,None].conjugate()
opt_SE = cal_SE(CSI_test, V_opt, noise)

# calculate ratio
RP2opt = np.mean(RP_SE) / np.mean(opt_SE) * 100

# print result
if __name__ == '__main__':
    print('RP SE:', np.mean(RP_SE))
    print('opt_SE:', np.mean(opt_SE))
    print('RP2opt:', np.round(RP2opt, 3), '%')

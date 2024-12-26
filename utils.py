
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split

# read data
FR = 1
BSloc = np.load(f'data/FR{FR}/BSloc.npy') # (1, 3)
UEloc = np.load(f'data/FR{FR}/UEloc.npy') # (N, 3)
CSI = np.load(f'data/FR{FR}/CSI.npy') # (N, Nr, Nt, Nc)
LoS = np.load(f'data/FR{FR}/LoS.npy') # (N,)
AoD = np.load(f'data/FR{FR}/AoD.npy') # (N,)

# process data
BSloc = BSloc[0, 0:2] # (2, )
UEloc = UEloc[:, 0:2] # (N, 2)
CSI = CSI.transpose([0,3,1,2]) # (N, Nc, Nr, Nt)
[N, Nc, Nr, Nt] = CSI.shape

# set communication parameters
c = 3e8
if FR == 1:
    f = 3.5e9
    B = 9.36e6
if FR == 2:
    f = 28e9 
    B = 95.04e6
lamda = c/f
d = 1/2*lamda

# eliminate block users
def eliminate_block(UEloc, CSI):

    '''
    input: UEloc (N, 2), CSI (N, Nc, Nr, Nt)
    output: UEloc (N_valid, 2), CSI (N_valid, Nc, Nr, Nt)

    '''
    index_valid = []

    for i in range(UEloc.shape[0]):
        if np.abs(np.sum(CSI[i,:])) > 1e-15:
            index_valid.append(i)

    UEloc = UEloc[index_valid,:]
    CSI = CSI[index_valid,:]
    return UEloc, CSI

# add position error to location
def add_error(loc, std):

    '''
    input: loc (N, 2) std: GPS standard deviation
    output: loc_error (N, 2)

    '''
    np.random.seed(1)
    error = np.random.normal(0, std/np.sqrt(2), loc.shape)
    loc_error = loc + error
    return loc_error

# normlize user location to [0,1]
def normalize_loc(loc):

    '''
    input: loc (N, 2)
    output: loc_norm (N, 2)

    '''
    loc_norm = np.empty(loc.shape)
    x_max = np.max(loc[:,0])
    x_min = np.min(loc[:,0])
    x_len = x_max - x_min
    loc_norm[:,0] = (loc[:,0] - x_min) / x_len
    y_max = np.max(loc[:,1])
    y_min = np.min(loc[:,1])
    y_len = y_max - y_min
    loc_norm[:,1] = (loc[:,1] - y_min) / y_len
    return loc_norm

# add Rayleigh fading to CSI
def add_fading(CSI, K):

    '''
    input: CSI (N, Nc, Nr, Nt), K: random scattering power ratiio
    output: CSI_fading (N, Nc, Nr, Nt)

    '''
    fading = np.empty(CSI.shape, dtype='complex')

    for i in range(CSI.shape[0]):
        CSI_power = (np.linalg.norm(CSI[i,:]))**2/np.prod(CSI[i,:].shape)
        fadding_std = np.sqrt(K * CSI_power / 2)
        np.random.seed(1)
        fading_real = np.random.normal(0, fadding_std, CSI[i,:].shape)
        np.random.seed(2)
        fading_imag = np.random.normal(0, fadding_std, CSI[i,:].shape)
        fading[i,:] = fading_real + 1j * fading_imag

    CSI_fading = CSI + fading
    return CSI_fading

# Add noise to CSI
def add_noise(CSI, SNR):

    '''
    input: CSI (N, Nc, Nr, Nt), SNR
    output: CSI_noise (N, Nc, Nr, Nt)

    '''
    power = (np.linalg.norm(CSI))**2/np.prod(CSI.shape)
    noise_std = np.sqrt(power / (10**(SNR/10)) / 2)
    np.random.seed(1)
    noise_real = np.random.normal(0, noise_std, CSI.shape)
    np.random.seed(2)
    noise_imag = np.random.normal(0, noise_std, CSI.shape)
    noise_comp = noise_real + 1j * noise_imag
    CSI_noise = CSI + noise_comp
    return CSI_noise

# normlize beamforming vector to meet power constraint
def normalize_V(V):

    '''
    input: V (N, Nc, Nt, 1)
    output: V_norm (N, Nc, Nt, 1)

    '''
    V_conj_tran = np.transpose(V, (0,1,3,2)).conjugate()
    V_power = np.matmul(V_conj_tran, V)
    V_norm = V / np.sqrt(V_power)
    return V_norm

# calcaulate spectral efficiency of all users
def cal_SE(H, V, noise):

    '''
    input: H (N, Nc, Nr, Nt), V (N, Nc, Nt, 1)
    output: rate_Nc

    '''
    HV = np.matmul(H, V)
    HV_gain = np.matmul(np.transpose(np.conj(HV), (0,1,3,2)), HV)
    HV_gain = np.squeeze(np.abs(HV_gain))
    SNR = HV_gain / noise
    rate = np.log2(1 + SNR)
    rate_Nc = np.mean(rate, axis=1) # average of subcarriers
    return rate_Nc

# compute the SE of LoS and NLoS conditions
def compute(rate_Nc):

    '''
    input: rate_Nc (N_test, )
    output: LoS_SE, NLoS_SE

    '''
    LoS = np.load(f'data/FR{FR}/LoS.npy') # (N,)
    LoS = LoS[LoS != -1]
    LoS_train, LoS_test = train_test_split(LoS, test_size=0.2, random_state=1)
    N_test = rate_Nc.shape[0]
    LoS_num, NLoS_num, LoS_SE, NLoS_SE = 0, 0, 0, 0
    
    for i in np.arange(N_test):
        if LoS_test[i] == 1:
            LoS_SE = LoS_SE + rate_Nc[i]
            LoS_num = LoS_num + 1
        if LoS_test[i] == 0:
            NLoS_SE = NLoS_SE + rate_Nc[i]
            NLoS_num = NLoS_num + 1
    
    LoS_SE = LoS_SE / LoS_num
    if NLoS_num == 0:
        NLoS_SE = 0
    else:
        NLoS_SE = NLoS_SE / NLoS_num
    return LoS_SE, NLoS_SE

# customized loss function
def su_loss(noise):
    def cust_loss(H, V):

        '''
        input: H (N, Nc, Nr, Nt), V (N, Nc, Nt, Nr)

        '''
        V_conj_tran = tf.transpose(V, (0,1,3,2), conjugate=True)
        power = tf.matmul(V_conj_tran, V)
        V_norm = V / tf.sqrt(power)
        HV = tf.matmul(H, V_norm)
        HV_conj = tf.transpose(HV, (0,1,3,2), conjugate=True)
        HV_gain = tf.abs(tf.matmul(HV_conj, HV))
        SNR = HV_gain / noise
        rate = tf.math.log(1 + SNR)/np.log(2)
        rate_Nc = tf.reduce_mean(rate, axis=1) # average of subcarriers
        rate_mean = tf.reduce_mean(rate_Nc) # average of users
        loss = - rate_mean
        return loss
    return cust_loss

# train
def train(x_train, y_train, x_test, y_test, model, noise, epoch):
    model.summary()
    model.compile(loss=su_loss(noise), optimizer=optimizers.Adam(learning_rate=1e-3))
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model/best.h5', monitor='val_loss',
                 verbose=1, save_best_only=True, mode='min', save_weights_only=False)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                patience=50, min_lr=1e-6, verbose=1)
    history = model.fit(x_train, y_train, epochs = epoch, batch_size = 128, verbose = 1,
              validation_split=0.5, callbacks=[checkpoint, reduce_lr])
    model.evaluate(x_test, y_test, verbose=1)
    model = models.load_model('model/best.h5', custom_objects={'cust_loss': su_loss(noise)})
    return model, history

# Geometry-based method
from utils import *
import math

# set parameters
loc_std = 1
fading_ratio = 0.1
SNR = 30

# process data
[UEloc, CSI] = eliminate_block(UEloc, CSI)
CSI = CSI / np.max(np.abs(CSI))
CSI_fading = add_fading(CSI, fading_ratio)

# divide train and test set
x_train, x_test, y_train, y_test = train_test_split(UEloc, CSI_fading, test_size=0.2, random_state=1)
N_test = x_test.shape[0]

# add position error to the test data
x_test = add_error(x_test, loc_std)

# calculate AoD based on locations
AoD = np.empty(N_test)
for i in np.arange(N_test):
    AoD[i] = math.atan((x_test[i,1] - BSloc[1]) / (x_test[i,0] - BSloc[0])) * 180 / np.pi

# calculate transmit precoding vector
V_GBM = np.empty([N_test, Nc, Nt, 1], dtype=complex)
for i in range(N_test):
    AoD[i] = AoD[i] * np.pi / 180
    for j in range(Nc):
        lamda = c / (f + j*B/Nc)
        for k in range(Nt):
            V_GBM[i,j,k] = np.exp(-1j*2*np.pi*k*d*np.sin(AoD[i])/lamda)

# normalize beamforming vector
V_GBM = normalize_V(V_GBM)

# noise
power = (np.linalg.norm(CSI_fading))**2/np.prod(CSI_fading.shape)
noise = power / (10**(SNR/10))

# calculate SE of GBM
GBM_SE = cal_SE(y_test, V_GBM, noise)

# calculate optimum
U, Sigma, VT = np.linalg.svd(y_test)
V_opt = VT[:,:,0,:][:,:,:,None].conjugate()
opt_SE = cal_SE(y_test, V_opt, noise)

# calculate ratio
GBM2opt = np.mean(GBM_SE) / np.mean(opt_SE) * 100

# calculate LoS and NLoS conditions
GBM_LoS_SE, GBM_NLoS_SE = compute(GBM_SE)
opt_LoS_SE, opt_NLoS_SE = compute(opt_SE)
GBM2opt_LoS = np.mean(GBM_LoS_SE) / np.mean(opt_LoS_SE) * 100
GBM2opt_NLoS = np.mean(GBM_NLoS_SE) / np.mean(opt_NLoS_SE) * 100

# print result
if __name__ == '__main__':
    print('GBM2opt:', np.round(GBM2opt, 3), '%')
    print('GBM LoS:', np.round(GBM2opt_LoS, 3), '%')
    print('GBM NLoS:', np.round(GBM2opt_NLoS, 3), '%')
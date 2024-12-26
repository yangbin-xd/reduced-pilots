
# channel knowledge map
from utils import *

# set parameters
loc_std = 5
fading_ratio = 0.1
SNR = 15

# read data
AoD = AoD[AoD != 0]
[UEloc, CSI] = eliminate_block(UEloc, CSI)
CSI = CSI / np.max(np.abs(CSI))
CSI_fading = add_fading(CSI, fading_ratio)

# divide train and test set
x_train, x_test, y_train, y_test = train_test_split(UEloc, CSI_fading, test_size=0.2, random_state=1)
AoD_train, AoD_test = train_test_split(AoD, test_size=0.2, random_state=1)
N_train, N_test = AoD_train.shape[0], AoD_test.shape[0]

# add position error to the test data
x_test = add_error(x_test, loc_std)

# calculate distance to apply inverse distance weighting (IDW)
dist = np.empty([N_test, N_train])
for i in range(N_test):
    for j in range(N_train):
        dist[i,j] = np.sqrt(np.sum((x_test[i] - x_train[j]) ** 2))

# K nearest neighbors (kNN) + IDW
def kNN(i, AoD_train, k):
    dist_sort = sorted(enumerate(dist[i,:]), key=lambda x:x[1])
    index = [x[0] for x in dist_sort]
    K = index[0:k]
    dist_k = dist[i,K]
    AoD_k = np.squeeze(AoD_train[K])
    weight = 1/dist_k
    weight = weight/np.sum(weight)
    AoD_IDW = np.dot(weight, AoD_k)
    return AoD_IDW

# calculate transmit precoding vector
V_CKM = np.empty([N_test, Nc, Nt, 1], dtype=complex)
for i in range(N_test):
    theta = kNN(i, AoD_train, 3) * np.pi / 180
    for j in range(Nc):
        lamda = c / (f + j*B/Nc)
        for k in range(Nt):
            V_CKM[i,j,k] = np.exp(-1j*2*np.pi*k*d*np.sin(theta)/lamda)

# normalize beamforming vector
V_CKM = normalize_V(V_CKM)

# noise
power = (np.linalg.norm(CSI_fading))**2/np.prod(CSI_fading.shape)
noise = power / (10**(SNR/10))

# calculate SE of CKM
CKM_SE = cal_SE(y_test, V_CKM, noise)

# calculate optimum
U, Sigma, VT = np.linalg.svd(y_test)
V_opt = VT[:,:,0,:][:,:,:,None].conjugate()
opt_SE = cal_SE(y_test, V_opt, noise)

# calculate ratio
CKM2opt = np.mean(CKM_SE) / np.mean(opt_SE) * 100

# calculate LoS and NLoS conditions
CKM_LoS_SE, CKM_NLoS_SE = compute(CKM_SE)
opt_LoS_SE, opt_NLoS_SE = compute(opt_SE)
CKM2opt_LoS = CKM_LoS_SE / opt_LoS_SE * 100
CKM2opt_NLoS = CKM_NLoS_SE / opt_NLoS_SE * 100

# print result
if __name__ == '__main__':
    print('CKM2opt:', np.round(CKM2opt, 3), '%')
    print('CKM LoS:', np.round(CKM2opt_LoS, 3), '%')
    print('CKM NLoS:', np.round(CKM2opt_NLoS, 3), '%')
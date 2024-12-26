
# extensions to multi-UE scenarios
from utils import *

# set parameters
SNR = 30

# local SNR or global SNR
if __name__ == '__main__':
    SNR = SNR
    np.save('data/SNR.npy', SNR)
else:
    SNR = np.load('data/SNR.npy')

# import data
from integrate_svm import V_svm_test
from reduced_pilot import CSI_fading, CSI_test

# noise
power = (np.linalg.norm(CSI_fading))**2/np.prod(CSI_fading.shape)
noise = power / (10**(SNR/10))

# for simplicity of user grouping
N_test = V_svm_test.shape[0]
N_test = int(N_test/8)*8
V_svm_test = V_svm_test[:N_test,:]
CSI_test = CSI_test[:N_test,:]

# calculate gain
def calculate_gain(H, V):

    '''
    input: H (Nc, Nr, Nt), V (Nc, Nt, Nr)

    '''
    HV = np.matmul(H, V)
    HV_conj = np.transpose(HV, (0,2,1)).conjugate()
    gain = np.squeeze(np.abs(tf.matmul(HV_conj, HV)))
    return gain

# perform pseudo inverse
def pseudo_inverse(V_SU, UE):

    '''
    input: V_SU (UE, Nc, Nt, Nr), UE
    output: V_MU (UE, Nc, Nt, Nr)

    '''
    V_MU = np.empty([Nc, Nt, UE], dtype='complex')
    for i in range(Nc):
        V_SU_k = np.squeeze(V_SU[:, i, :, :]).T
        V_SU_k_H = V_SU_k.T.conj()
        V_MU[i,:] = (np.linalg.inv(V_SU_k_H @ V_SU_k) @ V_SU_k_H).T.conj()
    V_MU = V_MU.transpose(2,0,1)[:,:,:,None]
    V_MU = normalize_V(V_MU)
    return V_MU

# calculate SE for multi-UE scenarios
def calculate_MUSE(H, V, UE):

    '''
    input: H (N, Nc, Nr, Nt), V (N, Nc, Nt, Nr)
    output: SUSE SU beamforming vector for multiple UE
            MUSE MU beamforming vector for multiple UE
    '''
    H = H.reshape([int(N_test/UE), UE, Nc, 1, Nt])
    V = V.reshape([int(N_test/UE), UE, Nc, Nt, 1])
    SUSE = np.empty([int(N_test/UE), UE])
    MUSE = np.empty([int(N_test/UE), UE])
    for i in range(int(N_test/UE)):
        H_SU = H[i,:]
        V_SU = V[i,:]
        V_MU = pseudo_inverse(V_SU, UE)
        V_SU = V_SU / np.sqrt(UE)
        V_MU = V_MU / np.sqrt(UE) 
        for j in range(UE):
            signal_SU = calculate_gain(H_SU[j,:], V_SU[j,:])
            signal_MU = calculate_gain(H_SU[j,:], V_MU[j,:])
            interference_SU = 0
            interference_MU = 0
            for k in range(UE):
                if k != j:
                    interference_SU += calculate_gain(H_SU[j,:], V_SU[k,:])
                    interference_MU += calculate_gain(H_SU[j,:], V_MU[k,:])
            SINR_SU = signal_SU / (interference_SU + noise)
            SINR_MU = signal_MU / (interference_MU + noise)
            SUSE[i,j] = np.mean(np.log2(1 + SINR_SU))
            MUSE[i,j] = np.mean(np.log2(1 + SINR_MU))
    return np.mean(np.sum(SUSE,axis=-1)), np.mean(np.sum(MUSE,axis=-1))

# calculate spectral efficiency of V_SU, V_MU for multi-UE scenarios
SUSE2, MUSE2 = calculate_MUSE(CSI_test, V_svm_test, 2)
SUSE4, MUSE4 = calculate_MUSE(CSI_test, V_svm_test, 4)
SUSE8, MUSE8 = calculate_MUSE(CSI_test, V_svm_test, 8)

# print result
if __name__ == '__main__':
    print('SU2:', np.round(SUSE2, 3))
    print('SU4:', np.round(SUSE4, 3))
    print('SU8:', np.round(SUSE8, 3))
    print('MU2:', np.round(MUSE2, 3))
    print('MU4:', np.round(MUSE4, 3))
    print('MU8:', np.round(MUSE8, 3))

# Generate deterministic propagation channel by DeepMIMO
# To run this file, please run $ pip install DeepMIMO
# Please see https://www.deepmimo.net/versions/v2-python/
# Then download the scenario O1 Blockage Scenario
# Please see https://www.deepmimo.net/scenarios/o1-blockage-scenario/
import DeepMIMO
import numpy as np
from pprint import pprint

# Load the default parameters
parameters = DeepMIMO.default_params()

# Change parameters for the setup
parameters['scenario'] = 'O1_3p5B' 
# parameters['scenario'] = 'O1_28B' 

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = r'scenarios'

# To only include 10 strongest paths in the channel computation
parameters['num_paths'] = 10

# To activate the user rows
parameters['user_row_first'] = 501
parameters['user_row_last'] = 1500

# To activate the selected rows randomly and the users in each selected row randomly
parameters['row_subsampling'] = 0.1
parameters['user_subsampling'] = 0.11

# Activate only the third basestation
parameters['active_BS'] = np.array([3]) 

# OFDM settings
if parameters['scenario'] == 'O1_3p5B':
    FR = 1
    parameters['OFDM']['bandwidth'] = 9.36e-3 # 624 * 15kHz = 9.36 MHz
    parameters['OFDM']['subcarriers'] = 52*12 # OFDM with 624 subcarriers
    parameters['OFDM']['subcarriers_limit'] = 12 # Keep only first 12 subcarriers
if parameters['scenario'] == 'O1_28B':
    FR = 2
    parameters['OFDM']['bandwidth'] = 95.04e-3 # 1584 * 60kHz = 95.04 MHz
    parameters['OFDM']['subcarriers'] = 132*12 # OFDM with 1584 subcarriers
    parameters['OFDM']['subcarriers_limit'] = 12 # Keep only first 12 subcarriers

# Antenna settings
parameters['ue_antenna']['shape'] = np.array([1, 1, 1]) # ULA of 1 elements
parameters['bs_antenna']['shape'] = np.array([1, 32, 1]) # ULA of 32 elements

# Generate and inspect the dataset
pprint(parameters)
dataset = DeepMIMO.generate_data(parameters)

# load data
BSloc = dataset[0]['basestation']['location']
UEloc = dataset[0]['user']['location']
CSI = dataset[0]['user']['channel']
LoS = dataset[0]['user']['LoS']

# Eliminate block users for AoD 
LoS_index = [i for i, x in enumerate(LoS) if x==1]
NLoS_index = [i for i, x in enumerate(LoS) if x==0]
Block_index = [i for i, x in enumerate(LoS) if x==-1]

index = np.array(LoS_index + NLoS_index)
index = np.array(sorted(index))

N = CSI.shape[0]
n = index.shape[0]

AoD = np.zeros(N)
for i in np.arange(n):
    AoD[index[i]] = dataset[0]['user']['paths'][index[i]]['DoD_phi'][0]

# save data
np.save(f'data/{FR}/BSloc.npy', BSloc)
np.save(f'data/{FR}/UEloc.npy', UEloc)
np.save(f'data/{FR}/CSI.npy', CSI)
np.save(f'data/{FR}/LoS.npy', LoS)
np.save(f'data/{FR}/AoD.npy', AoD)

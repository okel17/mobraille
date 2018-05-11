
import os 
TEST = './segmented_data_test/segblock24.jpg'
value = os.system('python classify_PCA.py ' + TEST)
os.system('python chain_coding.py ' + TEST)
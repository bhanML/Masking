# disturb the CIFAR-100 with the specific noise
import numpy as np

# binary format
# <1 byte for coarse label> <1 byte for fine label> <3072 byte for pixels>

def gen_noise_tridiagonal():
  # Three diagonals
  NOISY_PROPORTION = 0.6
  T = np.zeros((10,10))
  T[0][0],T[0][1] = 1.0 - NOISY_PROPORTION + 0.2, NOISY_PROPORTION - 0.2
  T[1][0],T[1][1],T[1][2] = NOISY_PROPORTION/2.0, 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0
  T[2][1],T[2][2],T[2][3] = NOISY_PROPORTION/2.0, 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0
  T[3][2],T[3][3],T[3][4] = NOISY_PROPORTION/2.0, 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0
  T[4][3],T[4][4],T[4][5] = NOISY_PROPORTION/2.0, 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0
  T[5][4],T[5][5],T[5][6] = NOISY_PROPORTION/2.0, 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0
  T[6][5],T[6][6],T[6][7] = NOISY_PROPORTION/2.0, 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0
  T[7][6],T[7][7],T[7][8] = NOISY_PROPORTION/2.0, 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0
  T[8][7],T[8][8],T[8][9] = NOISY_PROPORTION/2.0, 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0
  T[9][8],T[9][9] = NOISY_PROPORTION - 0.2, 1.0 - NOISY_PROPORTION + 0.2

  return T

def gen_noise_column():
  # Column disturbs
  NOISY_PROPORTION = 0.6
  T = np.zeros((10,10))
  T[0][0],T[0][3],T[0][5] = 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0, NOISY_PROPORTION/2.0
  T[1][1],T[1][3],T[1][5] = 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0, NOISY_PROPORTION/2.0
  T[2][2],T[2][3],T[2][5] = 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0, NOISY_PROPORTION/2.0
  T[3][3],T[3][5] = 1.0 - NOISY_PROPORTION + 0.2, NOISY_PROPORTION - 0.2
  T[4][4],T[4][3],T[4][5] = 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0, NOISY_PROPORTION/2.0
  T[5][5],T[5][3] = 1.0 - NOISY_PROPORTION + 0.2, NOISY_PROPORTION - 0.2
  T[6][6],T[6][3],T[6][5] = 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0, NOISY_PROPORTION/2.0
  T[7][7],T[7][3],T[7][5] = 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0, NOISY_PROPORTION/2.0
  T[8][8],T[8][3],T[8][5] = 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0, NOISY_PROPORTION/2.0
  T[9][9],T[9][3],T[9][5] = 1.0 - NOISY_PROPORTION, NOISY_PROPORTION/2.0, NOISY_PROPORTION/2.0

  return T

def dis_noise(label,T):
   return xrange(10)[np.argmax(np.random.multinomial(size=1,n=1,pvals=T[label]))]

def inject_data(noise, filename, noise_name):
   with open(filename,'rb') as f:
      with open(filename[:-4] + '_' + noise_name  + filename[-4:],'wb') as w:
        e = f.read(3073)
        while e:
          label = ord(e[0])
          #print(label)
          dis_label = dis_noise(label,noise)
          dis_e = chr(dis_label) + e[1:]
          w.write(dis_e)
          e = f.read(3073)



tridiagonal_noise = gen_noise_tridiagonal()
inject_data(tridiagonal_noise,'data/cifar10/cifar-10-batches-bin/data_batch_1.bin','tridiagonal')
inject_data(tridiagonal_noise,'data/cifar10/cifar-10-batches-bin/data_batch_2.bin','tridiagonal')
inject_data(tridiagonal_noise,'data/cifar10/cifar-10-batches-bin/data_batch_3.bin','tridiagonal')
inject_data(tridiagonal_noise,'data/cifar10/cifar-10-batches-bin/data_batch_4.bin','tridiagonal')
inject_data(tridiagonal_noise,'data/cifar10/cifar-10-batches-bin/data_batch_5.bin','tridiagonal')

column_noise = gen_noise_column()
inject_data(column_noise, 'data/cifar10/cifar-10-batches-bin/data_batch_1.bin','column')
inject_data(column_noise, 'data/cifar10/cifar-10-batches-bin/data_batch_2.bin','column')
inject_data(column_noise, 'data/cifar10/cifar-10-batches-bin/data_batch_3.bin','column')
inject_data(column_noise, 'data/cifar10/cifar-10-batches-bin/data_batch_4.bin','column')
inject_data(column_noise, 'data/cifar10/cifar-10-batches-bin/data_batch_5.bin','column')


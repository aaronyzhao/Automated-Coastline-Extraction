#  Reads .MAT color file

import scipy.io
mat = scipy.io.loadmat("data/color150.mat")
print(mat)
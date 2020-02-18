import numpy as np

# Define the given matrix
A = np.asarray([[-5,  -5,  -1,  0,  0,  0,  500,  500,  100],
                [0,  0,  0,  -5,  -5,  -1,  500,  500,  100],
                [-150,  -5,  -1,  0,  0,  0,  30000,  1000,  200],
                [0,  0,  0,  -150,  -5,  -1,  12000,  400,  80],
                [-150,  -150,  -1,  0,  0,  0,  33000,  33000,  220],
                [0,  0,  0,  -150,  -150,  -1,  12000,  12000,  80],
                [-5,  -150,  -1,  0,  0,  0,  500,  15000,  100],
                [0,  0,  0,  -5,  -150,  -1,  1000,  30000,  200]])
# Generate square matrices of the form 9x9 and 8x8 respectively
W1 = A.T.dot(A)
W2 = A.dot(A.T)

# Evaluate V
V = np.linalg.eig(W1)[1]
print('V:', V)

# Evaluate sigma
eig_val = np.absolute(np.linalg.eig(W1)[0])
s = np.sqrt(eig_val)
S_ = np.diag(s)
S = S_[0:8, :]
print('Sigma:', S)

# Evaluate U
U = np.linalg.eig(W2)[1]
print('U:', U)

# Evaluate H
H = V[:, 8]
H = np.reshape(H, (3, 3))
print('H:', H)

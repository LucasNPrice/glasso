import numpy as np


def block_matrix(inputmatrix, outersize, innersize):
  p = outersize; M = innersize
  innerblocks = []
  for row in range(0,p):
    for col in range(0,p):
      innerblocks.append(inputmatrix[row*M:row*M+M, col*M:col*M+M])
  blocked = np.asarray(innerblocks).reshape(p,p,M,M)
  return(blocked)


def block_diag(outersize, innersize):
  p = outersize; M = innersize
  innerblocks = []
  for row in range(0,p):
    for col in range(0,p):
      if row == col:
        innerblocks.append(np.identity(M))
      else:
        innerblocks.append(np.zeros((M,M)))
  blocked_diag = np.asarray(innerblocks).reshape(p,p,M,M)
  return(blocked_diag)
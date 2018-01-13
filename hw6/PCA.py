from skimage import transform
from skimage import io 
import sys
import numpy as np
import os

#io.use_plugin('gtk', 'imshow')
X = []
dir_name = sys.argv[1]

dim = 600
#print(os.listdir(dir_name))
for file in os.listdir(dir_name):
	#io.use_plugin('matplotlib', 'imshow')
	images = io.imread(dir_name+'/'+file)
	images = np.array(images)
	#images = images.astype('float64')
	#new_img = transform.resize(old_img, new_shape)
	#dim = images.shape[0]//3
	#images = transform.resize(images, (images.shape[0]//3, images.shape[1]//3, images.shape[2]))
	#io.imshow(tmp)
	X.append(np.array(images).reshape(-1))


#io.find_available_plugins(loaded=True);
X = np.array(X)
X = X.T
#print(X.shape)

X_mean = np.mean(X, 1)
"""
X_mean -= np.min(X_mean)
X_mean /= np.max(X_mean)
X_mean = (X_mean * 255).astype(np.uint8)
X_mean = X_mean.reshape((dim, dim, 3))
io.imsave('mean.jpg', X_mean)
"""
X_mean = X_mean.reshape((-1, 1))
#print('X_mean.shape=', X_mean.shape)
#y = X - X_mean
"""
input('stop!!')
print('mean!!')
input('stop!!')
"""
U, s, V = np.linalg.svd(X - X_mean, full_matrices=False)
"""
eigface = U[:,3]
print(U.shape)
print(eigface)
eigface -= np.min(eigface)
eigface /= np.max(eigface)
eigface = (eigface * 255).astype(np.uint8)
eigface = eigface.reshape((dim, dim, 3))
io.imsave('eigface3.jpg', eigface)

input('stop!!')
print('eigenface!!')
input('stop!!')
"""
k = 700
U = U[:,:k]
#print(U.shape)
#c = int(sys.argv[2].split('.jpg')[0])
#print('c=',c)
#print('X[:,c]')
#print(X[:,c])
#print('X_mean')
#print(X_mean)
#y = X[:,c].reshape((-1,1)) - X_mean
y = io.imread(dir_name+'/'+sys.argv[2])
y = np.array(y)
y = np.array(y).reshape((-1, 1))
y_tmp = np.array(y)
y = y - X_mean
#print('y')
#print(y)
w = []
for t in U.T:
	t = t.reshape((-1,1))
	tmp = np.dot(y.T, t)
	#print(tmp.shape)
	w.append(tmp)
w = np.array(w)
#print('w')
#print(w)
w = w.reshape(-1)
#print('w')
#print(w)
#w_sum = np.sum(w)
#w = w/w_sum
#U_w = w*U
U_w = np.sum(w*U, 1)
X_mean = X_mean.reshape(-1)
M = U_w + X_mean

M -= np.min(M)
M /= np.max(M)
M = (M * 255).astype(np.uint8)
M = M.reshape((dim, dim, 3))
io.imsave('reconstruction.jpg', M)

print(np.sqrt(np.mean((M.reshape(-1) - y_tmp.reshape(-1))**2)))

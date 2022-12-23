#importing  libraries

import numpy as np
from matplotlib import pyplot as plt

#define function

def covar(him):
    covar=np.dot(np.transpose(ri),(ri))/N
    
    return covar
   

def cor (him):
    u=(np.mean(np.transpose(ri), 1))
    cor=np.dot(np.transpose(ri-u),(ri-u))/N
    
    return cor


def mohalanobas_distance (d,him,Func):
    invF=np.linalg.inv(Func(him))
    #print(invF)
    mohalanobas_distance =np.zeros(N)#4096
    for p in range (N):
        dr=(d-ri[p,:])
        mohalanobas_distance[p] = dr@invF@np.transpose(dr)
    mohalanobas_distance =np.reshape(mohalanobas_distance,(him.shape[0],him.shape[1]))
    
    return  mohalanobas_distance

def cov_filt(d,him):
    invK = np.linalg.inv(cor(him))
    u=(np.mean(np.transpose(ri), 1))
    cov_filt = np.zeros(N)
    for p in range(N):
        du=(u-d)
        dr=(d-ri[p,:])
        cov_filt[p] = np.reshape(du,(1,169))@invK@np.reshape(dr,(169,1))
    cov_filt = np.reshape(cov_filt,(him.shape[0],him.shape[1]))
    
    return cov_filt

def cor_filt(d,him):
    invR = np.linalg.inv(covar(him))
    cor_filt = np.zeros(N)
    for p in range(N):
        cor_filt[p] = np.reshape(d,(1,169))@invR@np.reshape(ri[p,:],(169,1))
    cor_filt = np.reshape(cor_filt,(him.shape[0],him.shape[1]))
    
    return cor_filt



'''main function '''

filepath =  r"panel.npy"
data =np.load(filepath,allow_pickle=True)
item=data.item()
groundtruth = data.item().get('groundtruth')
him =np.array( data.item().get('HIM'),"double")
gtp = np.argwhere(groundtruth == 1)
N=him.shape[0]*him.shape[1] #64*64
ri=np.reshape(him,(N,him.shape[2]))

p1=him[7,37,:]
p2=him[20,35,:]
p3=him[34,34,:]
p4=him[47,33,:]
p5=him[59,33,:]





'''plot cmd'''

fig=plt.figure()
plt.title('CMD')
for i in range(1,6):  
    img = mohalanobas_distance(eval('p'+str(i)),him,cor)
    fig.add_subplot(1,5,i)
    plt.imshow(img, cmap= 'gray_r', interpolation = 'nearest')
plt.show()

'''plot RMD'''

fig=plt.figure()
plt.title('RMD')
for i in range(1,6):
    img =  mohalanobas_distance(eval('p'+str(i)),him,covar)
    fig.add_subplot(1,5,i)
    plt.imshow(img, cmap= 'hot_r', interpolation = 'nearest')
plt.show()

'''plot cmfm'''

fig=plt.figure()
plt.title('CMFM')
for i in range(1,6):
    img = cov_filt(eval('p'+str(i)),him)
    fig.add_subplot(1,5,i)
    plt.imshow(img, cmap= 'gray_r', interpolation = 'nearest')
plt.show()

'''plot rmfm'''

fig=plt.figure()
plt.title('RMFM')
for i in range(1,6):
    img = cor_filt(eval('p'+str(i)),him)
    fig.add_subplot(1,5,i)
    plt.imshow(img,  cmap= 'hot_r', interpolation = 'nearest')
plt.show()


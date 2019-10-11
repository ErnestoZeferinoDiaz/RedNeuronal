import numpy as np
import matplotlib.pyplot as plt


def lineal(x):
    return x

def dx_lineal(x):
    return 1

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def dx_sigmoid(x):
    return np.multiply((1-x),x)

def tanh(x):
    ep=np.exp(x)
    en=np.exp(-x)
    return (ep-en)/(ep+en)

def dx_tanh(x):
    return 1-np.multiply(x,x)    


data=np.genfromtxt('datos.csv',delimiter=',')

#X=np.matrix(data[:,0]).transpose()
#Y=np.matrix(data[:,1]).transpose()
X=np.matrix([[0,0],[0,1],[1,0],[1,1]])
Y=np.matrix([[-1],[1],[1],[-1]])
nn=[X.shape[1],30,Y.shape[1]]
min=-1
max=1

W1 = min+np.random.rand(nn[1],nn[0])*(max-min+1)
W2 = min+np.random.rand(nn[2],nn[1])*(max-min+1)
#W3 = min+np.random.rand(nn[3],nn[2])*(max-min+1)

b1 = min+np.random.rand(nn[1],1)*(max-min+1)
b2 = min+np.random.rand(nn[2],1)*(max-min+1)
#b3 = min+np.random.rand(nn[3],1)*(max-min+1)

alfa = 0.01

i=0
error_medio=[]
while(i<200000):
    j=0
    suma=0
    while(j<X.shape[0]):
        a0 = X[j,:].transpose()
        a1 = tanh(np.matmul(W1,a0) + b1)
        a2 = tanh(np.matmul(W2,a1) + b2)
        #a3 = lineal(np.matmul(W3,a2) + b3)

        e = Y[j,:].transpose() - a2
        
        suma = e**2 + suma
        
        #s3 = -2*dx_lineal(a3)*e        
        #s2 = np.diag(dx_tanh(a2))*W3.transpose()*s3
        s2 = -2*dx_tanh(a2)*e
        s1 = np.diag(dx_tanh(a1))*W2.transpose()*s2

        #W3 = np.copy(W3 - alfa*s3*a2.transpose())
        W2 = np.copy(W2 - alfa*s2*a1.transpose())
        W1 = np.copy(W1 - alfa*s1*a0.transpose())
        #b3 = np.copy(b3 - alfa*s3)
        b2 = np.copy(b2 - alfa*s2)
        b1 = np.copy(b1 - alfa*s1)
 
        j=j+1
    print((suma/j)[0,0])
    error_medio.append((suma/j)[0,0])
    i=i+1
plt.plot(range(len(error_medio)),error_medio,'-',linewidth=1,color=(0,0,0))
plt.show()

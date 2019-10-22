import numpy as np
import matplotlib.pyplot as plt

def lineal(x):
    return [x,1]

def sigmoid(x):
    return [1.0/(1.0+np.exp(-x)),x-np.power(x,2)]

def tanh(x):
    ep=np.exp(x)
    en=np.exp(-x)
    return [(ep-en)/(ep+en),1-np.power(x,2)]
    
data=np.genfromtxt('datosSeno.csv',delimiter=',')

X=np.matrix(data[:,0]).transpose()
Y=np.matrix(data[:,1]).transpose()

rows = X.shape[0]

red = [X.shape[1],30,40,Y.shape[1]]

minimo = -1
maximo = 1
#W1=np.matrix(np.genfromtxt('W1.csv',delimiter=',')).transpose()
#W2=np.matrix(np.genfromtxt('W2.csv',delimiter=','))
#W3=np.matrix(np.genfromtxt('W3.csv',delimiter=','))
#b1=np.matrix(np.genfromtxt('b1.csv',delimiter=',')).transpose()
#b2=np.matrix(np.genfromtxt('b2.csv',delimiter=',')).transpose()
#b3=np.matrix(np.genfromtxt('b3.csv',delimiter=',')).transpose()
W1 = minimo + np.random.rand(red[1],red[0]) * (maximo - minimo)
W2 = minimo + np.random.rand(red[2],red[1]) * (maximo - minimo)
W3 = minimo + np.random.rand(red[3],red[2]) * (maximo - minimo)
b1 = minimo + np.random.rand(red[1],1) * (maximo - minimo)
b2 = minimo + np.random.rand(red[2],1) * (maximo - minimo)
b3 = minimo + np.random.rand(red[3],1) * (maximo - minimo)

alfa = 0.1

emedio=[]
eI=1
epocas=0
while(eI>10**(-3)):
    suma=0
    for i in range(rows):
        a0 = X[i,:].transpose()
        z1 = np.dot(W1,a0) + b1
        a1 = sigmoid(z1)[0]
        z2 = np.dot(W2,a1) + b2
        a2 = sigmoid(z2)[0]
        z3 = np.dot(W3,a2) + b3
        a3 = lineal(z3)[0]     

        e = Y[i,:].transpose()-a3
        
        s3 = -2*(lineal(a3)[1])*e
        s2 = np.diagflat(sigmoid(a2)[1])*W3.transpose()*s3
        s1 = np.diagflat(sigmoid(a1)[1])*W2.transpose()*s2  
        
        W3 = W3 - alfa*s3*a2.transpose()
        W2 = W2 - alfa*s2*a1.transpose()
        W1 = W1 - alfa*s1*a0.transpose()
        
        b3 = b3 - alfa*s3        
        b2 = b2 - alfa*s2        
        b1 = b1 - alfa*s1  
        
        suma = e.transpose()*e + suma
        
    emedio.append((suma/rows)[0,0]) 
    eI=emedio[epocas]
    print(eI)
    epocas=epocas+1

a=[]
test = np.linspace(0,6,200)
for i in test:
    a0 = i
    z1 = np.dot(W1,a0) + b1
    a1 = sigmoid(z1)[0]
    z2 = np.dot(W2,a1) + b2
    a2 = sigmoid(z2)[0]
    z3 = np.dot(W3,a2) + b3
    a3 = lineal(z3)[0]     
    a.append(a3[0,0])

fig,graficas = plt.subplots(1,2)
graficas[0].plot(range(len(emedio)),emedio,"b-",linewidth=1)
graficas[1].plot(X,Y,"ro",linewidth=1)
graficas[1].plot(test,a,"b-",linewidth=1)
plt.show()













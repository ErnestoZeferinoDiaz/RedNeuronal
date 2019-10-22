% Autor: Erik Zamora G�mez
% Fecha 31/07/2015
% Este c�digo es distribuido bajo la licencia CC BY-NC-SA
clear all, close all, clc

R=csvread('datosSeno.csv')
X=R(:,1)
Y=R(:,2)
P=[0,0;
   0,1;
   1,0;
   1,1]
T=[0,1,1,0]' 
P=X;
T=Y;
Q = size(P,1);

red = [size(P,2),30,40,size(T,2)]
% Valores iniciales
ep = 1;
minimo = -1;
maximo = 1;
W1 = minimo + rand(red(2),red(1)) * (maximo - minimo);
W2 = minimo + rand(red(3),red(2)) * (maximo - minimo);
W3 = minimo + rand(red(4),red(3)) * (maximo - minimo);

b1 = minimo + rand(red(2),1) * (maximo - minimo);
b2 = minimo + rand(red(3),1) * (maximo - minimo);
b3 = minimo + rand(red(4),1) * (maximo - minimo);

W1=csvread('W1.csv');
W2=csvread('W2.csv');
W3=csvread('W3.csv');
b1=csvread('b1.csv');
b2=csvread('b2.csv');
b3=csvread('b3.csv');

alfa = 0.1;

eI=1;
Epocas=1;


while(eI>10^(-3))
    sum = 0;
    for q = 1:Q
        a0 = P(q,:)';
        z1=W1*a0 + b1;
        a1 = sigmoid(z1);
        z2=W2*a1 + b2;
        a2 = sigmoid(z2);
        z3 = W3*a2 + b3;
        a3 = z3;
        
        e = T(q)'-a3;
        
        s3 = -2*(1)*e;
        s2 = diag((1-a2).*a2)*W3'*s3;
        s1 = diag((1-a1).*a1)*W2'*s2;
        
        W3 = W3 - alfa*s3*a2';
        W2 = W2 - alfa*s2*a1';
        W1 = W1 - alfa*s1*a0';
        
        b3 = b3 - alfa*s3;        
        b2 = b2 - alfa*s2;        
        b1 = b1 - alfa*s1;  
        
        
        sum = e'*e + sum;
    end
    % Error cuadratico medio
    emedio(Epocas) = sum/Q;
    eI=emedio(Epocas)
    Epocas++;
end
figure
plot(emedio)

test = [0:0.01:6];
for q = 1:length(test)
    a0 = test(q);
    a1 = sigmoid(W1*a0 + b1);
    a2 = sigmoid(W2*a1 + b2);
    a3 = W3*a2 + b3;
    a(q)=a3;
end
figure
plot(test,a,P,T,'r*')

 
u = linspace(-2, 2, 100);
v = linspace(-2, 2, 100);
for i = 1:length(u)
    for j = 1:length(v)
        a0 = [u(i); v(j)];
        a1 = sigmoid(W1*a0 + b1);
        a2 = sigmoid(W2*a1 + b2);
        a3 = sigmoid(W3*a2 + b3);
        z(i,j) = a3;   
    end
end
figure
hold on
contour(u, v, z')  %Por que debo transponer a z
plot(P(find(T==0),1),P(find(T==0),2),"ro")
plot(P(find(T==1),1),P(find(T==1),2),"bo")
colorbar
axis([-0.5 1.5 -0.5 1.5])

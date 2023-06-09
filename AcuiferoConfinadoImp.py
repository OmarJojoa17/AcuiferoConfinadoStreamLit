# -*- coding: utf-8 -*-

#Packages required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title('Solución numérica por diferencias finitas para acuifero confinado')

st.latex(r'''\frac{∂ h^2}{∂ x^2} ± \frac{w}{T} = \frac{S}{T} \frac{∂ h}{∂ t}''')

#Fix values
st.header('Valores fijos')
L = 30 #length [m]
b = 10 #acuifer's power [m]
st.markdown('Longitud entre el acuifero y el río es: '+str(L)+' m.')
st.markdown('La potencia del acuífero es: '+str(b)+' m.')

#Values from streamlit
st.header('Valores seleccionados por el usuario')
K = st.slider('Seleccione el valor de conductividad hidráulica (K) [m/d]:', 0.00007,1.4,0.5, 0.001, format='%f')
T = K*b #Value assumed, mean value of Clay
Ss = st.slider('Seleccione el valor de coeficiente de almacenamiento (Ss) [1/m]:',0.000006,0.002,0.0011,0.00001, format='%f')
S = Ss*b #Storage Coeficient
Dh = T/S #Hydraulic Diffusivity
st.markdown('El valor de difusividad hidráulica es: '+str(round(Dh,2))+' d/m$^2$.')

#Discretize the problem
st.header('Discretización del problema numérico')
N = L+1 #Nodes
x = np.arange(0,L) #values in x
dx = L/(N-1) #Delta x
st.markdown('Número de nodos: '+str(N)+'.')
st.markdown('Intervalo de cálculo en x (dx) [m]: '+str(dx)+'.')

#Initial conditions
st.header('Condiciones iniciales y de borde del problema')
t = 0
hini = 8 #Total energy initial
H = np.ones(N)*hini #Whole nodes with the initial conditions t = 0
st.markdown('h(x,t=0): '+str(hini))
#Boundary conditions
h0 = 0
H[0] = h0
st.markdown('h(x=0,t): '+str(h0))
st.markdown('h(x=L,t): '+str(hini))

#Solution with t
st.header('Condiciones temporales del problema')
dt = st.slider('Seleccione el paso de tiempo (dt) [d]: ', 0.01,10.0,1.0,0.01, format='%f')

tfin = [200,400,600,800,1000,10000,100000,1000000,10000000] #Final time when the values achieve 0
Hsol = [H]
tsol = [t]

def acuifero_confinado(N:int, h0:int, Dh:float, dt:float, dx:float, Hsol:list, tsol:list, H, t:float, tfin:float)->pd.DataFrame:
    while t<tfin:
        #Loop for the system equation
        A = np.zeros((N,N)) #Matriz A
        b = np.zeros(N) #Vector with the results
        for i in range(N):
            #Node x=0
            if i==0: 
                A[i][i] = 1
                b[i] = h0
            #Node x=L
            elif i==N-1:
                A[i][i] = 1+Dh*dt/dx**2
                A[i][i-1] = -Dh*dt/dx**2
                b[i] = H[i]
            #Central nodes
            else:
                A[i][i] = 1+2*Dh*dt/dx**2
                A[i][i+1] = -Dh*dt/dx**2
                A[i][i-1] = -Dh*dt/dx**2
                b[i] = H[i]
        #Solution for the system
        Ainv = np.linalg.inv(A)
        Hdt = np.dot(Ainv,b)
        #Update data for next time   
        H = Hdt.copy()
        t=t+dt
        Hsol.append(H)
        tsol.append(t) 
    #Results
    Hsol = np.array(Hsol)
    tsol = np.round(np.array(tsol),2)
    return Hsol, tsol

#Find the result in the row that achieve value in h less than 0.00001
for i in range(len(tfin)):
    #Reset to initial conditions for each iteration
    t = 0
    hini = 8 #Total energy initial
    H = np.ones(N)*hini #Whole nodes with the initial conditions t = 0
    H[0] = 0
    Hsol = [H]
    tsol = [t]
    Hsol, tsol = acuifero_confinado(N, h0, Dh, dt, dx, Hsol, tsol, H, t, tfin[i])
    b = 0
    for j in range(Hsol.shape[0]):
        if Hsol[j][-1] < 0.001: #This is almost cero
            break
        else: 
            b += 1
    if b < tfin[i]:
        break

Hsol = Hsol[:b]
tsol = tsol[:b]
st.markdown('El tiempo para el cual el acuífero alcanza un equilibrio es: '+str(round(b*dt,0))+' días.')

Hsol2=pd.DataFrame(Hsol)
tsel = st.slider('Seleccione el tiempo para el cual desea ver el perfil de agua: ',0.1,float(b-1),float(b-1),float(dt), format='%f')
plt.plot(Hsol2.columns,Hsol[int(tsel)])
plt.ylim(0,8.1)
st.pyplot(plt)

st.markdown('Code developed by: Omar David Jojoa Ávila.')

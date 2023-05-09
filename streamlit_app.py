# -*- coding: utf-8 -*-

#First day
#import streamlit as st
#st.write('Hello world')

"""
#St.slider
import streamlit as st
from datetime import time, datetime

st.header('st.slider')

#Ejemplo 1
st.subheader('Slider')
age = st.slider('How old are you?', 0,130,25)
st.write("I'm ", age, 'years old')
         
#Ejemplo 2
st.subheader('Range slider')
values = st.slider('Select a range of values',
                   0.0, 100.0, (25.0, 75.0))
st.write('Values:', values)

#Example 3
st.subheader('Range time slider')
appointment = st.slider(
    "schedule your appointment:",
    value=(time(11,30), time(12,45)))
st.write("You're scheduled for:", appointment)

#Ejemplo 4
st.subheader('Datetime slider')
start_time = st.slider(
    "When do you start?",
    value = datetime(2020,1,1,9,30),
    format="MM/DD/YY - hh:mm")
st.write('Start time:', start_time)
"""

#st.line_chart: make lines with properties from the data
import streamlit as st
import pandas as pd
import numpy as np

st.header('Line chart')
chart_data = pd.DataFrame(
    np.random.rand(20,3),
    columns=['a','b','c'])
st.line_chart(chart_data)

st.latex(r'''\frac{∂ h^2}{∂ x^2} \pm w = \frac{S}{T} \frac{∂ h}{∂ t}''')

L = 30 #length [m]
b = 10 #acuifer's power [m]
st.markdown('Longitud entre el acuifero y el río es: '+str(L)+'m')
st.markdown('La potencia del acuífero es: '+str(b)+'m')

st.header('Valores seleccionados por el usuario:')
K = st.slider('Seleccione el valor de K [m/d]', 0.00007,1.4,1.0, 0.01, format='%f')
T = K*b #Value assumed, mean value of Clay
Ss = st.slider('Seleccione el valor de Ss [1/m]',0.000006,0.002,0.001,0.00001, format='%f')
S = Ss*b #Storage Coeficient


"""Things of the main script for streamlit"""
"""
Usign the ecuation from Craig's Soil Mechanics

#Boundary conditions and soil's parameters
ho = 8 #Boundary condition
b = 10 #Value assumed
K = 7.776E-5 #Value assumed of Hydraulic Conductivity (m/d), mean value of Clay
T = K*b #Value assumed, mean value of Clay
Ss = 1.0995E-3 #Value assumed od Sepcific Storage (1/m), mean value of Medium Hard clay
S = Ss*b #Storage Coeficient
L = 30  #Value assumed, but it should be given by the problem
Dh = S/T #Hydraulic diffusivity

#Infinite summatory
ms = 1000 #Value so long in order to replicate the infinite in summatory
Tv = 0 #Start value in 0
M = 0 #Start value in 0
ts = np.arange(0,200,1) #Delta time that I assign
dist = np.arange(0,L+1)#Delta x that I define, 2 by 2 steeps until 30m
tiempos = list() #Start times in 0

resul = pd.DataFrame() #Create a frame in order to save results
resul['tiempos (m)'] = ts
resul.set_index('tiempos (m)', inplace=True)

suma = 0
for x in dist:
    tiempos = list()
    for t in ts:
        suma = 0
        for m in range(ms):
            M = np.pi/2*(2*m+1)
            Tv = Dh*t/L**2
            suma += (2*ho/M)*np.sin(M*x/L)*np.exp(-(M**2)*Tv)
        tiempos.append(suma)
    resul[str(x)] = tiempos

#I already have the responses from the aquifer, then I wanna transpose the last matrix
#in order to graph the responses curves
df = resul.copy()
df = df.T

#Plot results in teha analytical for
plt.figure(figsize=(20,12))
plt.plot(df.index, df)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Solución Analítica Acuifero confinado', fontsize=25)
plt.xlabel('Distancia (m)', fontsize=20)
plt.ylabel('Altura potenciométrica (m)', fontsize=20)
plt.savefig(r'SoluciónAnalítica.jpg')
"""

""" Explicit method
#Data from the problem
L = 30 #length
b = 10 #Value assumed, but it should be given by the problem
K = 7.776E-5 #Value assumed of Hydraulic Conductivity (m/d), mean value of Clay
T = K*b #Value assumed, mean value of Clay
Ss = 1.0995E-3 #Value assumed od Sepcific Storage (1/m), mean value of Medium Hard clay
S = Ss*b #Storage Coeficient
Dh = S/T #Hydraulic Diffusivity

#Discretize the problem
N = L+1 #Nodes
x = np.arange(0,L) #values in x
dx = L/(N-1) #Delta x

#Initial conditions
t = 0
hini = 8 #Total energy initial
H = np.ones(N)*hini #Whole nodes with the initial conditions t = 0

#Boundary conditions
h0 = 0
H[0] = h0

#Stability criteria
dtestable = dx**2/(2*Dh)

#Solution with t
dt = 0.03 #Delta t that I deffine, change this number for more general values
tfin = 200 #Final time when the values achieve 0, change this number for more general values
Hdt= H.copy()
Hsol = [H]
tsol = [t]

w = 0 #Sink and 

while t<tfin:
    for i in range(N):
        if i==0:
            Hdt[i] = h0
        elif i==N-1: #Frontera con la conducción (caudal)
            Hdt[i] = ((Dh*dt/dx**2)*H[i-1]+
                     (1-Dh*dt/dx**2)*H[i])+dt*w
        else:
            Hdt[i] = ((Dh*dt/dx**2)*H[i-1]+
                     (1-2*Dh*dt/dx**2)*H[i]+
                     (Dh*dt/dx**2)*H[i+1])+dt*w
    H=Hdt.copy()
    t=t+dt
    Hsol.append(H)
    tsol.append(t)

#Results
Hsol = np.array(Hsol)
tsol = np.round(np.array(tsol),2)

#Plot results in the numercial form
Hsol2 = pd.DataFrame(Hsol)
plt.figure(figsize=(20,12))
for i in range(len(tsol)):
    plt.plot(Hsol2.columns, Hsol[i])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Solución Numérica (Diferencias Finitas) Acuifero confinado', fontsize=25)
    plt.xlabel('Distancia (m)', fontsize=20)
    plt.ylabel('Altura potenciométrica (m)', fontsize=20)
"""


""" All Implicit method. Finite Differences implicit method apllied to the statement

#Data from the problem
L = 30 #length
b = 10 #Value assumed, but it should be given by the problem
K = 7.776E-5 #Value assumed of Hydraulic Conductivity (m/d), mean value of Clay
T = K*b #Value assumed, mean value of Clay
Ss = 1.0995E-3 #Value assumed od Sepcific Storage (1/m), mean value of Medium Hard clay
S = Ss*b #Storage Coeficient
Dh = S/T #Hydraulic Diffusivity

#Discretize the problem
N = L+1 #Nodes
x = np.arange(0,L) #values in x
dx = L/(N-1) #Delta x

#Initial conditions
t = 0
hini = 8 #Total energy initial
H = np.ones(N)*hini #Whole nodes with the initial conditions t = 0

#Boundary conditions
h0 = 0
H[0] = h0

#Solution with t
dt = 1 #Delta t that I deffine
tfin = 200 #Final time when the values achieve 0
Hsol = [H]
tsol = [t]

#Assing boundary conditions Dirichlet (First) y Neuman (Second) 
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

#Plot results in the numercial form
Hsol2 = pd.DataFrame(Hsol)
plt.figure(figsize=(20,12))
for i in range(len(tsol)):
    plt.plot(df.index, Hsol[i])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Solución Numérica (Diferencias Finitas) Acuifero confinado', fontsize=25)
    plt.xlabel('Distancia (m)', fontsize=20)
    plt.ylabel('Altura potenciométrica (m)', fontsize=20)
plt.savefig(r'SoluciónNumérica.jpg')

#Animation
#In order to plot the animation put in the Console next text: %matpolotlib auto
fig = plt.figure(figsize=(20,12))
ax = plt.subplot(111)

#Function to make the animation
def actualizar(i):
    ax.clear()
    plt.plot(df.index,Hsol[i], color='blue', label='Solución Numérica')
    plt.plot(df.index,df[i], color='red', label='Solución Analítica')
    plt.title('Comparación solución analítica y numérica - t = '+str(i)+' días', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Distancia (m)', fontsize=20)
    plt.ylabel('Altura potenciométrica (m)', fontsize=20)
    plt.legend()
    plt.ylim(h0,hini+1)
    
import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, actualizar,range(len(tsol-1)))
plt.show()

#Export into GIF
#import networkx as nx
#from matplotlib.animation import PillowWriter
#ani.save(r'ComparacionSolucionAnaliticayNumerica.gif', writer=animation.PillowWriter(fps=3))
"""


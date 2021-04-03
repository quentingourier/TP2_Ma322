# -*- coding: utf-8 -*-
"""
@author: quentin gourier
@matiere: Ma322
@classe : 3SB2
"""

#Imports
import numpy as np
import matplotlib.pyplot as pp
import math as m
from scipy.integrate import odeint

#-------------------------------------------
#-------------------------------------------

#Initialisation des données
A = m.pi/2
g = 9.81
L = 1
phi = 0

#Pendule par teta
pas = 0.04
t = np.arange(0, 4+pas, pas)
teta = []
for i in t:
    teta.append(A*m.cos(m.sqrt(g/L)*i+phi))

#-------------------------------------------
#-------------------------------------------

#Fonctions

def pendule(Y,t):
    g = 9.81
    L = 1
    w = m.sqrt(g/L)
    #Calcul de Y'
    Yp = np.array([Y[1],-(w**2)*m.sin(Y[0])])
    return Yp

def Euler_Exp(f,y0,h):
    Ye =  np.zeros(shape=(len(t),2))
    y = y0
    for n in range (0,len(t)):
        Ye[n,0] = y[0]
        Ye[n,1] = y[1]
        y = y + h*f(y,t[n])
    return Ye

def Rung_Kutta(f,y0,h):
    Ye =  np.zeros(shape=(len(t),2))
    y = y0
    for n in range (0,len(t)):
        Ye[n,0] = y[0]
        Ye[n,1] = y[1]
        k1 = f(y,t[n])
        k2 = f(y+(h/2)*k1,t[n]+(h/2))
        k3 = f(y+(h/2)*k2,t[n]+(h/2))
        k4 = f(y+h*k3,t[n]+h)
        y = y + (h/6)*(k1+2*k2+2*k3+k4)
    return Ye

def suspension(Y,t):
    M1, M2 = 15, 200
    C2 = 1200
    K1, K2 = 50000, 5000
    f_t = -1000
    #Calcul de Y'
    x1 = Y[2]
    x2 = Y[3]
    x3 = -(C2/M1)*Y[2] + (C2/M1)*Y[3] -((K1+K2)/M1)*Y[0] + (K2/M1)*Y[1]
    x4 = (C2/M2)*Y[2] - (C2/M2)*Y[3] + (K2/M2)*Y[0] - (K2/M2)*Y[1] + (1/M2)*f_t
    Yp = np.array([x1, x2, x3, x4])
    return Yp    
    

#-------------------------------------------
#-------------------------------------------

#Partie 4 : Pendule
Y0 = np.array([m.pi/2,0])
sol1 = Euler_Exp(pendule,Y0,pas)
sol2 = Rung_Kutta(pendule,Y0,pas)
Yode = odeint(pendule, Y0, t)
equ_lin = pendule(Y0, t)

pp.figure()
pp.plot(t,teta, label = "teta(t)", color = 'fuchsia')
pp.plot(t,sol1[:,0],label="Euler explicite", color = 'sandybrown')
pp.plot(t,sol2[:,0],label="Runge_Kutta", color = 'silver', linewidth = 5)
pp.plot(t,Yode[:,0],label="Odeint", color = 'r')
pp.title("Tracés des différentes méthodes de résolution d'équation différentielle")
pp.xlabel('temps')
pp.ylabel('valeurs des solutions associées')
pp.legend()
pp.show()


#Partie 5 : Bonus 
pp.figure()
pp.plot(sol1.T[0], sol1.T[1],label="Euler explicite", color = 'sandybrown')  
pp.plot(sol2.T[0], sol2.T[1],label="Runge_Kutta", color = 'silver', linewidth = 5) 
pp.plot(Yode.T[0], Yode.T[1],label="Odeint", color = 'r') 
pp.title("Portrait de phase pour chaque méthode de résolution")
pp.xlabel("teta")
pp.ylabel("teta'")
pp.legend()
pp.show()

#Partie 6 : Suspension
Y0 = np.array([0, 0, 0, 0])
t = np.arange(0, 3, 0.03)
susp = odeint(suspension, Y0, t)
pp.figure()
pp.title("Résolution numérique : suspension d'un véhicule")
pp.plot(t, susp.T[0], label = "x1(t)", color = 'chartreuse')
pp.plot(t, susp.T[1], label = "x2(t)", color = 'darkblue')
pp.xlabel('temps')
pp.ylabel('tracé de la suspension')
pp.legend()
pp.show()





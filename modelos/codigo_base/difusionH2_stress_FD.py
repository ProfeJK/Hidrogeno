#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 08:19:53 2024
Diffusion cylinder implicit FD constant diffusivity
@author: whadymacbook2016
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

Ro=2.0e-2
Ri=Ro-3.0e-3
Cin=0.1
Cout=0.0
D=1.0e-8
E=10e9 #Pa
nu=0.3
Omega=5.0e-3
C0=0.0
pin=400.0e3 #Pa https://www.ugc.edu.co/pages/juridica/documentos/institucionales/NTC_2505_Instalaciones_Suministro_De_Gas.pdf
n=2000 #nodes
dr=(Ro-Ri)/(n-1)
#dt=0.083
#S=31536000 # seconds in a year
S=60*60 # seconds per hour
t_end=5.0
#nt=int(t_end/dt)
nt=1000
dt=t_end/(nt-1)


Cold=np.zeros(n)
C=np.zeros(n)
Disp=np.zeros(n)
sigma_r=np.zeros(n)
sigma_t=np.zeros(n)
epsi_r=np.zeros(n)
epsi_t=np.zeros(n)
A=np.zeros((3,n))
ADisp=np.zeros((5,n))
rhs=np.zeros(n)
rhsDisp=np.zeros(n)
r=np.linspace(Ri,Ro,n)
t=np.linspace(0,nt*dt,nt+1)
H=np.zeros((n,nt+1))
Cflux=np.zeros(n)
Hflux=np.zeros((n,nt+1))
HDisp=np.zeros((n,nt+1))
HStress_r=np.zeros((n,nt+1))
HStress_t=np.zeros((n,nt+1))
HStrain_r=np.zeros((n,nt+1))
HStrain_t=np.zeros((n,nt+1))

Cold[:]=C0
H[:,0]=Cold
Hflux[:,0]=0

# modelo de difusion y matrices para ambos problemas
for i in range(1,n-1):
    A[1+i-i,i]=-2*S*D/dr**2-1/dt
    A[1+i-(i+1),i+1]=S*D/dr**2+S*D/(2*r[i]*dr)
    A[1+i-(i-1),i-1]=S*D/dr**2-S*D/(2*r[i]*dr)
    ADisp[2+i-(i+1),i+1]=1/dr**2+1/(2*r[i]*dr)
    ADisp[2+i-(i-1),i-1]=1/dr**2-1/(2*r[i]*dr)
    ADisp[2+i-i,i]=-2/dr**2-1./r[i]**2
A[1+0-0,0]=1 
A[1+n-1-(n-1),n-1]=1
ADisp[2+0-0,0]=-3*(nu-1.)/(2*dr)-nu/r[0]
ADisp[2+0-1,1]=4*(nu-1.)/(2*dr)
ADisp[2+0-2,2]=-(nu-1.)/(2*dr)
ADisp[2+n-1-(n-3),n-3]=(nu-1.)/(2*dr)
ADisp[2+n-1-(n-2),n-2]=-4*(nu-1.)/(2*dr)
ADisp[2+n-1-(n-1),n-1]=3*(nu-1.)/(2*dr)-nu/r[n-1]


for j in range(nt):
    for i in range(1,n-1):
        rhs[i]=-Cold[i]/dt
    rhs[0]=Cin
    rhs[n-1]=Cout
    C=solve_banded((1,1),A,rhs)
    for i in range(1,n-1): # post processing de flujos
        Cflux[i]=-2.0*np.pi*r[i]*D*(C[i+1]-C[i-1])/(2*dr)
    Cflux[0]=-2.0*np.pi*r[0]*D*(-3*C[0]+4*C[1]-C[2])/(2*dr)  
    Cflux[n-1]=-2.0*np.pi*r[n-1]*D*(3*C[n-1]-4*C[n-2]+C[n-3])/(2*dr)
    H[:,j+1]=C
    Hflux[:,j+1]=Cflux
    Cold[:]=C
    
# Resolver desplazamientos debido a la concentración de hidrógeno
for i in range(1, n - 1):
    rhsDisp[i] = C[i] * (1 - nu) / ((1 + nu) * E)
U = solve_banded((1, 1), ADisp, rhsDisp)

# Calcular esfuerzos radial (Sr) y tangencial (St)
for i in range(1, n - 1):
    du_dr = (U[i + 1] - U[i - 1]) / (2 * dr)
    u_r = U[i] / r[i]
    Sr[i] = E / (1 - nu**2) * (du_dr + nu * u_r)
    St[i] = E / (1 - nu**2) * (u_r + nu * du_dr)

#post processing de flujos totales    
tot_flux_in=0
tot_flux_out=0
for i in range(nt):
    tot_flux_in+=0.5*(Hflux[0,i]+Hflux[0,i+1])*S*dt
    tot_flux_out+=0.5*(Hflux[n-1,i]+Hflux[n-1,i+1])*S*dt

# modelo de esfuerzos    
for j in range(nt+1):
    C[:]=H[:,j]
    for i in range(1,n-1):
        rhsDisp[i]=-(1.0/3.0)*(Omega/(nu-1.0))*(C[i+1]-C[i-1])/(2.*dr)
    rhsDisp[0]=-pin*(1.+nu)*(2*nu-1.)/E-(1./3.)*Omega*C[0]
    rhsDisp[n-1]=-(1./3.)*Omega*C[n-1]
    Disp=solve_banded((2,2),ADisp,rhsDisp)
    HDisp[:,j]=Disp
#post processing de stress and strain    
    epsi_t=Disp/r
    for i in range(n):
        if i>0 and i<n-1:
            epsi_r[i]=(Disp[i+1]-Disp[i-1])/(2*dr)
        if i==0:
            epsi_r[i]=(-3*Disp[i]+4*Disp[i+1]-Disp[i+2] )/(2*dr)
        if i==n-1:    
            epsi_r[i]=(3*Disp[i]-4*Disp[i-1]+Disp[i-2] )/(2*dr)
        sigma_r[i]=(E/((1.+nu)*(2.*nu-1.)))*\
        ((nu-1.)*epsi_r[i]-nu*epsi_t[i]+(1./3.)*Omega*C[i])
        sigma_t[i]=(E/((1.+nu)*(2.*nu-1.)))*\
        ((nu-1.)*epsi_t[i]-nu*epsi_r[i]+(1./3.)*Omega*C[i])
    HStress_r[:,j]=sigma_r
    HStress_t[:,j]=sigma_t
    HStrain_r[:,j]=epsi_r
    HStrain_t[:,j]=epsi_t

    
#grafica Concentracion vs r para cada tiempo    
plt.figure()
for i in range(0,nt+1,1):
    plt.plot(r*1e3,H[:,i],label=f't={i*dt:.1f} years') 
plt.xlabel('r [mm]')
plt.ylabel('C') 
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
#grafica concentracion vs tiempo para cada radio
plt.figure()
for i in range(0,n,10):
    plt.plot(t[:],H[i,:],label=f'r={i*dr*1e3:.1f} mm') 
plt.xlabel('t [years]')
plt.ylabel('C')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

#grafica flujo vs r para cada tiempo
plt.figure()
for i in range(0,nt+1,1):
    plt.plot(r*1e3,Hflux[:,i],label=f't={i*dt:.1f} years') 
plt.xlabel('r [mm]')
plt.ylabel('Flux $m^3/s$') 
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

#grafica flujos en radio interno y externo vs tiempo
plt.figure()
plt.plot(t,Hflux[0,:],label=f'inner radius flow') 
plt.plot(t,Hflux[n-1,:],label=f'outer radius flow') 
plt.xlabel('t [years]')
plt.ylabel('Flux $m^3/s$') 
plt.legend(loc='upper right')

#grafica desplazamiento vs radio para cada tiempo
plt.figure()
for i in range(0,nt+1,20):
    plt.plot(r*1e3,HDisp[:,i]*1e3,label=f't={i*dt:.1f} years') 
plt.xlabel('r [mm]')
plt.ylabel('u [mm]') 
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

#grafica sigma_r vs radio para cada tiempo
plt.figure()
for i in range(0,nt+1,1):
    plt.plot(r*1e3,HStress_r[:,i],label=f't={i*dt:.1f} years') 
plt.xlabel('r [mm]')
plt.ylabel('$\sigma_r$ [Pa]') 
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

#grafica sigma_t vs radio para cada tiempo
plt.figure()
for i in range(0,nt+1,1):
    plt.plot(r*1e3,HStress_t[:,i],label=f't={i*dt:.1f} years') 
plt.xlabel('r [mm]')
plt.ylabel(r'$\sigma_\theta$ [Pa]')  
#https://stackoverflow.com/questions/10370760/matplotlib-axis-label-theta-does-not-work-theta-does
#If you specify that the string is raw text (a r before the quotation mark), it works
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

#grafica epsilon_r vs radio para cada tiempo
plt.figure()
for i in range(0,nt+1,20):
    plt.plot(r*1e3,HStrain_r[:,i],label=f't={i*dt:.1f} years') 
plt.xlabel('r [mm]')
plt.ylabel(r'$\epsilon_r$') 
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

#grafica epsilon_t vs radio para cada tiempo
plt.figure()
for i in range(0,nt+1,20):
    plt.plot(r*1e3,HStrain_t[:,i],label=f't={i*dt:.1f} years') 
plt.xlabel('r [mm]')
plt.ylabel(r'$\epsilon_\theta$') 
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


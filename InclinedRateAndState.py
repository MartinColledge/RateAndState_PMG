#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 22:17:18 2021

@author: martin
"""

"""
Implementation of rate-and-state friction on an incline with horizontal pull
CORRECTION: NOT INCLINED HERE TO SIMPLIFY THE PROBLEM
"""

# Import usual modules

import numpy as np
import matplotlib.pyplot as plt


Dc      = 10e-6             # Critical slip distance (m)
m       = 5e5               # Mass (kg)
g       = 9.81              # Gravity acceleration (m.s-2)
b       = 0.1               # Rate and State b value
a       = 0.075             # Rate and state a value
k       = 0.1*(m*g)*(b-a)/Dc   # Spring stiffness (Pa) taken to be 0.1*kcrit as defined in Seagall 2010
Shear   = 30e9              # Shear modulus (Pa)
Vs      = 3000              # Shear wave velocity (m.s-1)
eta     = Shear/2*Vs        # Radiation damping factor (Pa.s.m-1)
Vplate  = 1e-9                 # Plate velocity (m.s-1)
V0      = 1                 # Arbitrary Normalizing Velocity (m.s-1)

Phi0=1                      # Initial value for Phi
Theta0=1                    # Initial value for Theta

#Delta_t=2000000        # Normalized Time Step (TO-DO, make it variable.)
   
def F1(Theta_t1,Phi_t1,Theta_t,Delta_t):
    # Residuals of the approximation for Theta
    return Theta_t1-Theta_t+Delta_t*(np.exp(-Theta_t1)-np.exp(Phi_t1))


def F2(Theta_t1,Phi_t1,Phi_t,Delta_t):
    # Residuals of the approximation for Phi
    return Phi_t1-Phi_t+Delta_t*(((Dc*k)/(m*g))*(Vplate/V0-np.exp(Phi_t1))-b*(np.exp(-Theta_t1)-np.exp(Phi_t1)))/(a+(V0*eta*np.exp(Phi_t1))/(m*g))    


def newton_raphson(Theta0,Phi0):
    # Set the variable time step as 1/exp(Phi)=V0/V, the larger the velocity, the smaller the time step
    Delta_t=100/np.exp(Phi0) 
    #First guess, the values are similar to those one step prior
    Theta_t1=Theta0
    Phi_t1=Phi0
    #Initialize storage of final guess at each time step
    Thetalist_NR=[]
    Philist_NR=[]
    #Threshold for implementation of iterative time step TODO properly
    Threshold=0.1
    #Initialize a guess that doens't match threshold and gets the while loop running
    Xn1=np.array([Theta0+1,Phi0+1])
    while np.linalg.norm(Xn1-np.array([Theta_t1,Phi_t1]))>Threshold:
        v=(a+(V0*eta*np.exp(Phi_t1))/(m*g))
        Jacobien = np.array([[1-Delta_t*np.exp(-Theta_t1), 
                              -Delta_t*np.exp(Phi_t1)  ], 
                            [ Delta_t*np.exp(-Theta_t1)*b/v,
                            1+ Delta_t/v**2*(((-Dc*k)/(m*g)*np.exp(Phi_t1)+b*np.exp(Phi_t1))*v \
                            -((Dc*k)/(m*g)*(Vplate/V0-np.exp(Phi_t1))-b*(np.exp(-Theta_t1)-np.exp(Phi_t1)))*((V0*eta)/(m*g)*np.exp(Phi_t1)))]
                            ])
        
        #Improve the approximation
        Xn1=np.array([[Theta_t1],[Phi_t1]])-np.dot(np.linalg.inv(Jacobien),np.array([[F1(Theta_t1,Phi_t1,Theta0,Delta_t)],[F2(Theta_t1,Phi_t1,Phi0,Delta_t)]]))
        #Separate the approximation elements for readability
        Theta_t1    = float(Xn1[0])
        Phi_t1      = float(Xn1[1])
        #Store the approximations
        Thetalist_NR.append(Theta_t1)
        Philist_NR.append(Phi_t1)
        #If the while loop has been through too many iterations then break
        if len(Philist_NR)>200:
            break
    #print(Jacobien)
    #Return the last approximation, and the list of prior approximations  (just in case, not really needed)
    return Xn1, Thetalist_NR, Philist_NR, Delta_t

def main():
    #Initialize the starting values
    Philist=[Phi0]
    Thetalist=[Theta0]
    Delta_tlist=[100/np.exp(Phi0)]
    #Iterate over a certain number of time steps
    n_iterations =100
    for i in range(0,n_iterations):
        Xn1,Thetalist_NR,Philist_NR,Delta_t_NR=newton_raphson(Thetalist[-1],Philist[-1])
        #Store the approximations at each time step
        Philist.append(Philist_NR[-1])
        Thetalist.append(Thetalist_NR[-1])
        Delta_tlist.append(Delta_t_NR)
    #Plot some values
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(np.cumsum(Delta_tlist),[a*Philist[i]+b*Thetalist[i] for i in range(0,n_iterations+1)])
    plt.xlabel('Time')
    plt.ylabel('a*log(V/V0))+log(V0*Theta/Dc) : Friction')
    plt.subplot(1,2,2)
    plt.plot(np.cumsum(Delta_tlist),[V0*np.exp(Philist[i]) for i in range(0,n_iterations+1)])
    plt.xlabel('Time')
    plt.ylabel('Velocity V')
    plt.show()


if __name__ == "__main__":
    main()
    
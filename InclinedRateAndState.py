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


Dc      = 10e-6 # Critical slip distance (m)
m       = 5e5   # Mass (kg)
g       = 9.81  # Gravity acceleration (m.s-2)
k       = 5e9   # Spring stiffness (Pa)
Shear   = 30e9  # Shear modulus (Pa)
Vs      = 3000   # Shear wave velocity (m.s-1)
eta     = Shear/2*Vs # Radiation damping factor (Pa.s.m-1)
Vplate  = 1e-8  # Plate velocity (m.s-1)
V0      = 1     # Arbitrary Normalizing Velocity (m.s-1)
b       = 0.1  # Rate and State b value
a       = 0.01 # Rate and state a value

Phi0=1          # Initial guess for Phi
Theta0=0.1      # Initial guess for Theta

Deltat=200000        # Normalized Time Step (TO-DO, make it variable.)
   
def F1(Theta_t1,Phi_t1,Theta_t):
    # Residuals of the approximation for Theta
    return Theta_t1-Theta_t+Deltat*(np.exp(-Theta_t1)-np.exp(Phi_t1))

def F2(Theta_t1,Phi_t1,Phi_t):
    # Residuals of the approximation for Phi
    return Phi_t1-Phi_t+Deltat*(((Dc*k)/(m*g))*(Vplate/V0-np.exp(Phi_t))-b*(np.exp(-Theta_t1)-np.exp(Phi_t1)))/(a+(V0*eta)/(m*g))    
  

def newton_raphson(Theta0,Phi0):
    #First guess, the values are similar to those one step prior
    Theta_t1=Theta0
    Phi_t1=Phi0
    #Initialize storage of final guess at each time step
    Thetalist=[]
    Philist=[]
    #Threshold for implementation of iterative time step TODO properly
    Threshold=0.1
    #Initialize a guess that doens't match threshold and gets the while loop running
    Xn1=np.array([Theta0+1,Phi0+1])
    while np.linalg.norm(Xn1-np.array([Theta_t1,Phi_t1]))>Threshold:
        
        Jacobien = np.array([[1-Deltat*np.exp(-Theta_t1), 
                              -Deltat*np.exp(Phi_t1)  ], 
                            [ Deltat*np.exp(-Theta_t1)*b/(a+(V0*eta)/(m*g)),
                            1+Deltat/(a+(V0*eta)/(m*g))*((-Dc*k)/(m*g)*np.exp(Phi_t1)+b*np.exp(Phi_t1))]
                            ])
        #Improve the approximation
        Xn1=np.array([[Theta_t1],[Phi_t1]])-np.dot(np.linalg.inv(Jacobien),np.array([[F1(Theta_t1,Phi_t1,Theta0)],[F2(Theta_t1,Phi_t1,Phi0)]]))
        #Seperate the approximation elements for readability
        Theta_t1    = float(Xn1[0])
        Phi_t1      = float(Xn1[1])
        #Store the approximations
        Thetalist.append(Theta_t1)
        Philist.append(Phi_t1)
        #If the while loop has been through too many iterations then break
        if len(Philist)>100:
            break
    #Return the last approximation, and the list of prior approximations  (just in case, not really needed)
    return Xn1, Thetalist, Philist

def main():
    #Initialize the starting values
    Philist=[Phi0]
    Thetalist=[Theta0]
    #Iterate over a certain number of time steps
    for i in range(0,1000):
        Xn1,Thetalist_NR,Philist_NR=newton_raphson(Thetalist[-1],Philist[-1])
        #Store the approximations at each time step
        Philist.append(Philist_NR[-1])
        Thetalist.append(Thetalist_NR[-1])
        
    #Plot some values
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(Thetalist)
    plt.xlabel('Time')
    plt.ylabel('log(V0*Theta/Dc)')
    plt.subplot(1,2,2)
    plt.plot(Philist)
    plt.xlabel('Time')
    plt.ylabel('log(V/V0)')
    plt.show()
    


if __name__ == "__main__":
    main()
    
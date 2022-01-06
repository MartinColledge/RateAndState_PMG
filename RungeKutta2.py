#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:56:40 2022

@author: martin
"""

def RK2(Delta_t,    # The time step
        f,          # The derivate of Theta that takes as arguments the values of Theta, u and Phi
        g,          # The derivate of u     that takes as argument  the value  of              Phi
        Theta_n,    # The initial value of                                        Thetha
        u_n,        # The initial value of                                               u
        Phi_n,      # The initial value of                                                     Phi
        Phi_n05     # The estimated value of Phi at midway throught the time step
        ):
    
    # Estimate Theta at the midway point with a forward discretization scheme
    Theta_n05   = Theta_n   + 0.5 * Delta_t * f(Theta_n,u_n,Phi_n)
    
    #Estimate u at the midway point in the same manner
    u_n05       = u_n       + 0.5 * Delta_t * g(Phi_n)
    
    #Estimate u at n+1
    u_n1        = u_n       +       Delta_t * g(Phi_n05)
    
    # Estimate Theta at n+1
    Theta_n1 = Theta_n + Delta_t * f(Theta_n05,u_n05,Phi_n05)
        
    return Theta_n1, u_n1
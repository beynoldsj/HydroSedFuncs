'''
This module contains functions for empirical fluvial geomorphology relationships.
Ben Reynolds, 2026
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# -------------------------------------------------------------------------------------------------
# region Gary Parker's bankfull geometry dataset and functions

class BankfullGeometry:
    '''
    Holder for the four variables involved in bankfull geometry to pass around. The inputs
    can be single values or numpy arrays.
    '''
    def __init__(self,Q,B,H,S,g=9.81):
        self.Q = Q 
        self.B = B 
        self.H = H 
        self.S = S
        self.g = g

        self.V = Q/(B*H)
        self.Fr = self.V / (g*H)**0.5

def ParkerRiverGeom(param_in, param='Q_bf'): 
    '''
    Parker Ebook relationships between bankfull discharge, Q_bf and:
    B_bf: bankfull width (m) 
    H_bf: bankfull depth (m) 
    S: slope

    note: the ranger of real rivers is +/- about 1/3 of an order of mag for width and depth,
    and about +/- a full order of magnitude for slope. The explantion is that the cross section 
    can readjust to a new quasi steady state much faster than the slope can
    '''

    match param:
        case "Q_bf":
            Q_bf = param_in
        case "B_bf":
            Q_bf = (param_in/2.8736)**(1/0.5788)
        case "H_bf":
            Q_bf = (param_in/0.3403)**(1/0.3627)
        case "S":
            Q_bf = (param_in/0.0115)**(1/-0.492)
        case _:
            raise ValueError("param options are Q_bf, B_bf, H_bf, and S")
    
    
    B_bf = 2.8736 * Q_bf**0.5788 
    H_bf = 0.3403 * Q_bf**0.3627 
    S = 0.0115 * Q_bf**-0.492 
    bankfull = BankfullGeometry(Q_bf,B_bf,H_bf,S)
    return bankfull


def GetParkerData(parkerGeomFi='./ShieldsJHRData.csv'):
    '''
    Read in Gary Parker's river bankfull geometry dataset
    '''

    riverGeom = pd.read_csv(parkerGeomFi)
    riverGeom['log D50 mm'] = np.log10(riverGeom['D50 mm'])
    riverGeom['Ubf ms'] = riverGeom['Qbf cms'] / (riverGeom['Bbf m'] * riverGeom['Hbf m'])
    riverGeom['Cz'] = riverGeom['Ubf ms'] / np.sqrt(9.81*riverGeom['Hbf m']*riverGeom['S'])
    return riverGeom


def PlotParkerRiverGeom(riverGeom,bankfull=None):
    '''
    Plot the typical relationships. Also plot a point on each of bankfull is handed in.

    riverGeom: river geometry dataframe as read in by GetParkerData 

    '''

    Q_bf_fit = np.linspace(2e-1,3e5,1000)
    BF = ParkerRiverGeom(Q_bf_fit)
    B_bf_fit = BF.B
    H_bf_fit = BF.H 
    S_fit = BF.S

    fig, ax = plt.subplots(2,2,figsize=(8,8),layout='constrained')
    ax[0,0].loglog() 
    ax[0,0].scatter(riverGeom['Qbf cms'], riverGeom['Hbf m'],c=np.log10(riverGeom['D50 mm'])) 
    ax[0,0].loglog(Q_bf_fit,H_bf_fit,'k--')
    ax[0,0].set_ylabel(r'$H_{bf} \: (m)$')
    #ax[0,0].plot(Q_bf_geom,H_bf,'ro',markersize=10,linewidth=10)
    ax[0,0].set_title('Depth vs dischrage')
    ax[0,0].set_xlabel(r'$Q_{bf} \: (m^3 s^{-1})$')

    ax[0,1].loglog() 
    ax[0,1].scatter(riverGeom['Qbf cms'], riverGeom['Bbf m'],c=np.log10(riverGeom['D50 mm'])) 
    ax[0,1].loglog(Q_bf_fit,B_bf_fit,'k--')
    ax[0,1].set_ylabel(r'$B_{bf} \: (m)$')
    #ax[0,1].plot(Q_bf_geom,B_bf,'ro',markersize=10,linewidth=10)
    ax[0,1].set_title('Width vs discharge')
    ax[0,1].set_xlabel(r'$Q_{bf} \: (m^3 s^{-1})$')

    ax[1,0].loglog() 
    ax[1,0].scatter(riverGeom['Qbf cms'], riverGeom['S'],c=np.log10(riverGeom['D50 mm'])) 
    ax[1,0].loglog(Q_bf_fit,S_fit,'k--')
    ax[1,0].set_ylabel(r'$S$')
    #ax[1,0].plot(Q_bf_geom,S,'ro',markersize=10,linewidth=10)
    ax[1,0].set_title('Slope vs discharge')
    ax[1,0].set_xlabel(r'$Q_{bf} \: (m^3 s^{-1})$')

    ax[1,1].loglog() 
    scatter1 = ax[1,1].scatter(riverGeom['Hbf m'], riverGeom['S'],c=np.log10(riverGeom['D50 mm'])) 
    ax[1,1].loglog(H_bf_fit,S_fit,'k--')
    ax[1,1].set_ylabel(r'$S$')
    #ax[1,1].plot(H_bf,S,'ro',markersize=10,linewidth=10)
    ax[1,1].set_title('Slope vs depth')
    ax[1,1].set_xlabel(r'$H_{bf} \: (m)$')

    fig.colorbar(scatter1, ax=[ax[1,1],ax[1,0]], shrink=0.6, location='bottom', label=r'$D50 \: log \: mm$')

    if bankfull is not None:
        ax[0,0].plot(bankfull.Q,bankfull.H,'ro',markersize=10,linewidth=10)
        ax[0,1].plot(bankfull.Q,bankfull.B,'ro',markersize=10,linewidth=10)
        ax[1,0].plot(bankfull.Q,bankfull.S,'ro',markersize=10,linewidth=10)
        ax[1,1].plot(bankfull.H,bankfull.S,'ro',markersize=10,linewidth=10)


    return fig, ax

def AddGeometryPoint(bankfull,ax,color='k'):
    ''' Function to add more points.
    
    '''
    ax[0,0].plot(bankfull.Q,bankfull.H,'o',markersize=10,linewidth=10,color=color)
    ax[0,1].plot(bankfull.Q,bankfull.B,'o',markersize=10,linewidth=10,color=color)
    ax[1,0].plot(bankfull.Q,bankfull.S,'o',markersize=10,linewidth=10,color=color)
    ax[1,1].plot(bankfull.H,bankfull.S,'o',markersize=10,linewidth=10,color=color)


def PlotParkerRiverGeom_V_Fr(riverGeom):
    '''Plot the bankfull velocity and Froude number vs bankfull width.
    
    '''

    Q_bf_fit = np.linspace(2e-1,3e5,1000)
    BF = ParkerRiverGeom(Q_bf_fit)

    riverGeom['Fr'] = riverGeom['Ubf ms'] / np.sqrt(BF.g*riverGeom['Hbf m'])

    fig, ax = plt.subplots(1,2,figsize=(9,4))

    ax[0].scatter(riverGeom['Hbf m'], riverGeom['Ubf ms'])
    ax[0].plot(BF.H,BF.V,'k--')
    ax[0].set_xscale('log',base=10)
    ax[0].set_xlabel(r'$H_{bf} \: (m)$') 
    ax[0].set_ylabel(r'$V_{bf} \: (m/s)$')
    ax[0].set_title('Velocity')

    ax[1].scatter(riverGeom['Hbf m'], riverGeom['Fr'])
    ax[1].plot(BF.H,BF.Fr,'k--')
    ax[1].set_xscale('log',base=10)
    ax[1].set_xlabel(r'$H_{bf} \: (m)$') 
    ax[1].set_ylabel(r'$Fr$')
    ax[1].set_title('Froude number')

    return fig, ax


# endregion

# -------------------------------------------------------------------------------------------------
# region Empirical planform discriminants and helpful functions

def BraidThreshold(Q_bf):
    '''
    Rhodes (2020) figure 8.3 based on Leopold and Wolman (1957)
    '''
    
    S_th = 0.0125*Q_bf**-0.44
    return S_th 

def calcPotentialPower(Q,S_v,D50,rho=1000.,g=9.81):
    '''
    potential specific stream power as in Kleinhans and van den Berg (2011) and earlier papers
    uses discharge (m^3/s) and divides by a reference width based on river type (sand/gravel)

    '''

    alpha = 4.7*np.ones(np.shape(Q)) # alpha for sand  
    alpha[D50>=2] = 3.0 # alpha for gravel 
    W_r = alpha * np.sqrt(Q)

    omega_pv = (rho*g*Q*S_v) / W_r 
    
    return omega_pv 

def SpecificStreamPower(U,H,S,rho_w=1000,g=9.81): 
    '''
    stream power per unit length and width 
    '''
    
    omega = rho_w*g*U*H*S
    
    return omega 


# endregion


# -------------------------------------------------------------------------------------------------
# region Typical scalings regarding meandering river kinematics

def tau_cut(tau_w,W_over_H):
    '''Meandering river cutoff timescale from Geyman et al.(2025) Scaling laws for sediment storage
    and turnover in floodplains. 
    
    tau_w = W/M_avg where W is width and M_avg = average meander migration rate. Time to migrate
        1 channel width. 100 is the base estimate (1% rule). Varies from ~10 to ~1000 years with
        far tails at 1 year and 10000 years. 
    W_over_H = W/H (no way) where W is width and H is depth. Can use parker emprical fits above 
        to estiamte.
    '''
    A = 810 / (W_over_H)
    return A*tau_w
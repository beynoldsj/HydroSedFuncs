import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import minimize_scalar

# -------------------------------------------------------------------------------------------------
# region BASIC HYDRAULIC FUNCTIONS

def ManningStrickler(y,D_x,alpha_s,alpha_r=8.1):
    '''Manning strickler relation for Cz as function of particle diameter. Ref ASCE 110 p27-31.
    Returning the dimensionless Chezy number preferred by Parker.

    D_x: is some percentile particle size. 
    alpha_s: multiplier from diameter to roughness element (k_s). Ref Table 2-1
    alpha_r: multiplier from 110. 8.1 is common. 8.32 is what Wright and Parker (2004) use...
    '''
    k_s = D_x*alpha_s
    Cz = alpha_r * (y/k_s)**(1./6.) # FIXME this is technically only for wide channels. Could use page 107 in Sturm
    return Cz 

def ChezyFromManning(H,n,K_n=1.0,g=9.81):
    '''Dimensionless Chezy coefficient from Manning's n
    
    n: manning's n 
    H: depth of flow 
    K_n: factor for non-homog units. 1.0 if using SI units. 1.49 if using ft and seconds 
    g: grave accel 
    '''
    Cz = (K_n/n) * H**(1/6) * g**(-1/2)
    return Cz

def ManningFromChezy(H,Cz,K_n=1.0,g=9.81):
    '''Manning's n from dimensionless Chezy
    
    Cz: Chezy friction coefficient
    H: depth of flow 
    K_n: factor for non-homog units. 1.0 if using SI units. 1.49 if using ft and seconds 
    g: grave accel 
    '''
    n = (K_n/Cz) * H**(1/6) * g**(-1/2)
    return n

def KinematicViscosityFromTemp(T): 
    '''Kinematic viscosity of water from temperature per ASCE 110 equation 2-46h
    
    T: temperature in C!
    '''
    nu = 1.79e-6 / (1 + .03368*T + .00021*T**2)  # m^2/s
    return nu 

def SpecificEnergy(y,V,g=9.81):
    '''Specific energy
    
    y: depth
    V: velocity
    '''
    # FIXME should is this only right for rectangular and trapezoidal channels bc centroid in center?
    return y + V**2/(2.*g)

def EnergySlope(n,V,R):
    return (n**2.) * (V**2.) / (R**(4/3))

# endregion 

# -------------------------------------------------------------------------------------------------
# region BASIC SEDIMENT TRANSPORT FUNCTIONS

def ReynoldsParticle(D,nu,rho_w=1000.0,rho_s=2650.0,g=9.81):
    # calculate the Reynolds particle number

    R = (rho_s/rho_w-1)
    Re_p = np.sqrt(R*g*D)*D/nu 
    
    return Re_p

def TermVelMadsen(D_S,A=0.954,B=5.12,D_S_to_D_N=0.9,nu=1.e-6,rho_w=1000.,rho_s=2650.,g=9.81):
    '''MADSEN (2006) empirical settling velocity with constants for worn quartz sand as default input
    as in ASCE 110 

    '''

    R = rho_s/rho_w-1.0 
    D_N = D_S / D_S_to_D_N

    S_star = D_N/(4.*nu) * np.sqrt(g*R*D_N) 
    v_s = np.sqrt(g*R*D_N)*(A+B/S_star)**-1.

    return v_s

def RouseFits(P,ref_depth):
    '''Provides the J1 and J2 values for Rouse profiles integrals (Ref ASCE110 Equation 2-221)
    
    P: Rouse number
    '''
    # empirical fits are ASCE110 2-221A and 2-221B with coefficients from the table that goes with em
    if ref_depth == .05:
        J1 = 1. / (1.1038 + 2.6626*P + 5.6497*P**2. + .3822*P**3. - .6174*P**4. + .1315*P**5. - .0091*P**6.)
        J2 = -1. / (1.2574 + 2.3159*P + 1.9239*P**2. - .3558*P**3. + .0075*P**4. +  .0064* P**5. - .0006*P**6)
    elif ref_depth == .1:
        J1 = 1. / (1.1266 + 2.6239*P + 3.0838*P**2. + -.3636*P**3. + -.0734*P**4. + .0246*P**5. + -.0019*P**6.)
        J2 = -1. / (1.4952 + 2.2041*P + 1.0552*P**2. + -.2372*P**3. + .0265*P**4. +  -.0008*P**5. + -.00005*P**6)
    else:
        raise ValueError('Error: RouseFits only setup for references of .05 or .1')

    return J1, J2

def RouseProfile(z,C_a,P,H,ref_frac):
    '''Calculate rouse profile for a given reference height as fraction of total depth
    
    z: single height or array of heights above bed
    P: Rouse number
    C_a: concentration (m^3/m^3)
    H: flow depth
    ref_frac: the fraction of total height where the concentration is known
    '''
    
    a = ref_frac*H
    C = C_a * ( ((H-z)/z) / ((H-a)/a) )**P
    return C 

def AverageConcentration(z,C_a,P,H,ref_frac):
    '''Calculate the average concentration for some range of z.
    
    z: single height or array of heights above bed
    P: Rouse number
    C_a: concentration (m^3/m^3)
    H: flow depth
    ref_frac: the fraction of total height where the concentration is known
    '''

    C = RouseProfile(z,C_a,P,H,ref_frac)
    C_bar = np.mean(C)
    return C_bar


# endregion


# -------------------------------------------------------------------------------------------------
# region STRESS PARTITION LOOKUP TABLE GENERATORS 

def solve_Hs_Sf_Wright_Parker(H,V,D,D_to_ks=2,alpha_r=8.32,R=1.65,g=9.81):
    '''Implements an iterative solver to get the two unkowns (H_s, flow depth corresponding to 
    the skin friction; S_f, the friction slope) using 2 equation.
    
    H: total flow depth (m) 
    V: flow velocity (m/s)
    D: In this case should be D50. FIXME would need to think about how to handle multiple classes 
    D_to_ks: multiplier to get roughness element size. See ASCE110 table FIXME 
    alpha_r: coefficient in manning strickler. 8.32 is used in the paper 
    R: specific grav
    g: grave accel (m/s^2)
    '''
    q_w = V*H
    H_s_guess = 0.4*H 
    k_s = D*D_to_ks
    def func(H_s):
        Cz_s = alpha_r*(H_s/k_s)**(1/6)
        S_f = (q_w/(H*Cz_s))**2 / (g*H_s)
        tau_star_s = (H_s*S_f)/(R*D)
        tau_star = (H*S_f)/(R*D)
        Fr = V / np.sqrt(g*H) 
        #return 0.05 + 0.7*(tau_star*Fr**0.7)**0.8 - tau_star_s 
        return np.abs(0.05 + 0.7*(tau_star*Fr**0.7)**0.8 - tau_star_s)
    
    solve_out = minimize(func,H_s_guess,method='nelder-mead',options={'disp':False})
    H_s_solve = solve_out.x[0]

    return H_s_solve 


def Cz_skin_drag_Wright_Parker(D50,H_range=[.1,10.1,.1],V_range=[.1,4.01,.01],
                               D_to_ks=2,alpha_r=8.32,R=1.65,g=9.81,make_plots=True):
    '''Creates lookup tables for the Wright Parker partition 
    
    D50: median grain size (mm) 
    H_range: [start, stop, increment] for flow depth (m)
    V_range: [start, stop, increment] for depth-avg velocity (m/s)
    D_to_ks: multiplier to get roughness element size. See ASCE110 table FIXME 
    alpha_r: coefficient in manning strickler. 8.32 is used in the paper 
    R: specific grav
    g: grave accel (m/s^2)
    '''

    k_s = D50*D_to_ks

    H_arr = np.arange(H_range[0],H_range[1],H_range[2])
    V_arr = np.arange(V_range[0],V_range[1],V_range[2])

    n_H = len(H_arr)
    n_V = len(V_arr)

    H_mesh, V_mesh = np.meshgrid(H_arr,V_arr)

    H_s_mesh = np.zeros_like(H_mesh)
    Fr_mesh = np.zeros_like(H_mesh)
    tau_star_s_mesh= np.zeros_like(H_mesh)
    tau_star_mesh = np.zeros_like(H_mesh)
    Cz_s_mesh = np.zeros_like(H_mesh)
    Cz_mesh = np.zeros_like(H_mesh)


    for ii in range(0,n_V):
        for jj in range(0,n_H):
            #print(V)

            V = V_mesh[ii,jj]
            H = H_mesh[ii,jj]

            q_w = V*H 
            Fr = V / np.sqrt(g*H)

            H_s = solve_Hs_Sf_Wright_Parker(
                H,V,D50,D_to_ks=D_to_ks,alpha_r=alpha_r,R=R,g=g
                )    
            Cz_s = alpha_r*(H_s/k_s)**(1/6)
            S_f = (q_w/(H*Cz_s))**2 / (g*H_s)
            tau_star_s = (H_s*S_f)/(R*D50)
            tau_star = (H*S_f)/(R*D50)
            u_star = np.sqrt(tau_star*R*g*D50)
            Cz = V/u_star 


            Fr_mesh[ii,jj] = Fr
            H_s_mesh[ii,jj] = H_s
            tau_star_s_mesh[ii,jj] = tau_star_s 
            tau_star_mesh[ii,jj] = tau_star
            Cz_s_mesh[ii,jj] = Cz_s 
            Cz_mesh[ii,jj] = Cz


    mask = tau_star_s_mesh<0.1

    Cz_plot = Cz_mesh.copy() 
    Cz_plot[mask] = np.nan

    tau_star_s_plot = tau_star_s_mesh.copy()
    tau_star_s_plot[mask] = np.nan

    tau_star_plot = tau_star_mesh.copy() 
    tau_star_plot[mask] =np.nan

    tau_star_ratio_mesh = tau_star_s_mesh / tau_star_mesh
    tau_star_ratio_plot = tau_star_ratio_mesh.copy()
    tau_star_ratio_plot[mask] = np.nan

    Fr_plot = Fr_mesh.copy()
    Fr_plot[mask] = np.nan

    if make_plots:
        fig, ax = plt.subplots(3,3,figsize=(12,8),layout='constrained')
        #ax[0,0].plot(V_mesh,Cz_plot,'k-')
        ax[0,0].scatter(V_mesh,Cz_plot,c=H_mesh,cmap=cmc.cm.batlow)
        ax[0,0].set_title('Dimensionless Chezy Number')
        ax[0,0].set_xlabel(r'$V \quad (m s^{-1})$')
        ax[0,0].set_ylabel(r'$Cz$')

        ax[0,1].scatter(V_mesh,tau_star_s_plot,c=H_mesh,cmap=cmc.cm.batlow)
        ax[0,1].set_title('Skin drag shields stress')
        ax[0,1].set_ylabel(r'$\tau_s^*$')
        ax[0,1].set_xlabel(r'$V \quad (m s^{-1})$')

        mappable = ax[0,2].scatter(V_mesh,tau_star_ratio_plot,c=H_mesh,cmap=cmc.cm.batlow)
        fig.colorbar(mappable)
        ax[0,2].set_xlabel(r'$V \quad (m s^{-1})$')
        ax[0,2].set_ylabel(r'$\tau_s^* / \tau^*$')
        ax[0,2].set_title('Friction shields stress to total')

        mappable = ax[1,0].pcolormesh(H_mesh,V_mesh,Cz_plot,cmap=cmc.cm.imola)
        ax[1,0].set_xlabel(r'$H \quad (m)$')
        ax[1,0].set_ylabel(r'$V \quad (m s^{-1})$')
        cbar = fig.colorbar(mappable)
        cbar.set_label(r'$Cz$')

        mappable = ax[1,1].pcolormesh(H_mesh,V_mesh,tau_star_s_plot,cmap=cmc.cm.hawaii)
        ax[1,1].set_xlabel(r'$H \quad (m)$')
        ax[1,1].set_ylabel(r'$V \quad (m s^{-1})$')
        cbar = fig.colorbar(mappable)
        cbar.set_label(r'$\tau_s^*$')

        mappable = ax[1,2].pcolormesh(H_mesh,V_mesh,tau_star_ratio_plot,cmap=cmc.cm.glasgow)
        ax[1,2].set_xlabel(r'$H \quad (m)$')
        ax[1,2].set_ylabel(r'$V \quad (m s^{-1})$')
        cbar = fig.colorbar(mappable)
        cbar.set_label(r'$\tau_s^* / \tau^*$')

        ax[2,0].plot(tau_star_plot*Fr_plot**.7,tau_star_s_plot,'.')
        ax[2,0].set_xlabel(r'$\tau^* Fr^{0.7}$')
        ax[2,0].set_ylabel(r'$\tau_s^*$')

        plt.show()

    return Cz_plot, tau_star_s_plot


# endregion


# -------------------------------------------------------------------------------------------------
# region SEDIMENT TRANSPORT METHODS 

class EmpiricalEngelundHansen:
    '''Gary Parker's Caltech Lecture 7 Slide 11'''

    def __init__(self,D50,rho_w=1000.,rho_s=2650.,g=9.81,T=20.):
        self.D50 = D50 
        self.rho_w = rho_w 
        self.rho_s = rho_s 
        self.g = g

        self.R = rho_s/rho_w-1.0
        self.nu = KinematicViscosityFromTemp(T)


    def Cz(self,y):
        R_p = np.sqrt(self.R*self.g*self.D50)*self.D50/self.nu 
        Cz = 0.144*R_p**0.197 * (y/self.D50)**0.413 
        return Cz 

    def q_s(self,y,V,alpha=0.05):
        Cz = self.Cz(y)
        v_star = V/Cz # shear velocity 
        tau_b = v_star**2 * self.rho_w
        tau_star = tau_b / (self.rho_w*self.R*self.g*self.D50)
        q_s_star = alpha * (Cz**2.) * (tau_star**2.5) 
        q_s = q_s_star * np.sqrt(self.R*self.g*self.D50) * self.D50 
        return q_s 
    
    def normalFlowWide(self,y,S,alpha=0.05):
        Cz = self.Cz(y)
        tau_b = self.rho_w*self.g*y*S
        v_star = np.sqrt(tau_b/self.rho_w)
        V = Cz*v_star
        q = V*y # water discharge per width
        q_s = self.q_s(y,V,alpha=alpha) # sediment discharge (m^3/s/m)
        
        results_out = {'U': V, 'q': q, 'q_t': q_s}
        return results_out 


    

class WrightParkerPlusDeLeeuw:
    '''Combined functions for the hyrdaulic resistance and the entrainment rate. 
    Using the partitioning between skin and form drag from Wright and Parker (2004b).
    Then using the entrainment relationship from DeLeeuw et al. (2020). 
    
    D50: sediment diameter. Though just thinking about one size currently. 
    rho_w=1000: water density
    rho_s=2650: sediment density
    lamb=0.3: porosity of sediment in active layer
    g=9.81: gravity 
    T=20.0: Temperature (C)
    
    Note that the partitioning setup is out of its typical application. Usually, a known stage 
    and velocity would give a basal stress and thus shields stress. Then the partitioning 
    would be applied to determine the shields stress that comes from skin friction, having been
    empirically fit to capture the partitioning. Here, for solving the gradually varied flow
    equations, the flow depth and velocity are known, so skin friction can be calculated via 
    Manning-Strickler. Then the partitioning is used to calculate the total shields stress 
    which can then give the energy loss required for incrementing the gradually varied flow
    solution. 
    '''

    def __init__(self,D50,rho_w=1000.,rho_s=2650.,lamb=0.3,g=9.81,T=20.):
        self.D50 = D50 
        self.rho_w = rho_w 
        self.rho_s = rho_s
        self.lamb = lamb 
        self.g = g
        self.T = T
        self.a_ref = 0.1 # reference depth (as fraction of full depth) for rouse profile. This relation uses 10%
        
        self.R = rho_s/rho_w-1.0 # specific gravity
        self.nu = KinematicViscosityFromTemp(T)

        # settling velocity from Fergusion and Church (2004) as used in de Leeuw et al. (2020)
        self.v_s = self.R*g*D50**2 / ( 18*self.nu + (0.75*1*self.R*g*D50**3.)**0.5 ) 

    
    def Cz(self,y,V,Fr):
        '''Take in stage and velocity as known during GVF steps. Calculate the dimensionless Chezy
        coefficient and the shear velocity corresponding ONLY to the skin friction.

        # y: stage 
        # V: average velocity
        # Fr: Froude number         
        '''
        # FIXME should use hyrdraulic radius with calc from Sturm.
        Cz_sk = ManningStrickler(y,self.D50,3.0) # FIXME should be using the height corresponding to skin drag, which we don't know

        u_star_sk = V/Cz_sk
        tau_b_sk = u_star_sk**2. * self.rho_w 
        tau_star_sk = tau_b_sk / (self.rho_w*self.R*self.g*self.D50) 
        tau_star = ( (tau_star_sk-.05)/(0.7*Fr**0.56) )**1.25 # rearanged wright and parker (asce 110 eq 2-177)
        tau_b = tau_star * (self.rho_w*self.R*self.g*self.D50) 
        u_star = np.sqrt(tau_b / self.rho_w)
        Cz = V/u_star 
        entrainment_inputs = {'u star sk': u_star_sk, 'Fr': Fr}
        return Cz, entrainment_inputs

    # # TAKE 2 - ees crap FIXME
    # def Cz(self,y,V,Fr,y_sk_guess=None):
    #     '''Take in stage and velocity as known during GVF steps. Calculate the dimensionless Chezy
    #     coefficient and the shear velocity corresponding ONLY to the skin friction.

    #     # y: stage 
    #     # V: average velocity
    #     # Fr: Froude number         

    #     # FIXME should make y_sk_guess and optional paramater that can be ignored
    #     '''
    #     # FIXME should use hyrdraulic radius with calc from Sturm.
    #     if y_sk_guess is None:
    #         y_sk_guess = 0.5*y 

    #     def f(y_sk):
    #         Cz_sk = ManningStrickler(y_sk,self.D50,3.0) # fixed with iterative solve :(
    #         S_f = (V/Cz_sk)**2 / (self.g*y_sk)
    #         tau_star_sk = (y_sk*S_f) / (self.R*self.D50) # jk this is tau_star
    #         tau_star = (y*S_f) / (self.R*self.D50)
    #         print(0.05 + 0.7*( tau_star * (V/np.sqrt(self.g*y))**0.7 )**0.8 - tau_star_sk)
    #         return np.abs(0.05 + 0.7*( tau_star * (V/np.sqrt(self.g*y))**0.7 )**0.8 - tau_star_sk)

    #     bounds = Bounds([0],[y[0]])
    #     solve_out = minimize(f,y_sk_guess,method='Nelder-Mead',bounds=bounds)
    #     print(solve_out)
    #     y_sk_solved = solve_out.x[0]
    #     print('y_sk_solved', y_sk_solved)

    #     #solve_out = fsolve(f,y_sk_guess)
    #     #y_sk_solved = solve_out[0]

    #     Cz_sk = ManningStrickler(y_sk_solved,self.D50,3.0) # fixed courtesy of solver above

    #     u_star_sk = V/Cz_sk
    #     tau_b_sk = u_star_sk**2. * self.rho_w 
    #     tau_star_sk = tau_b_sk / (self.rho_w*self.R*self.g*self.D50) 
    #     tau_star = ( (tau_star_sk-.05)/(0.7*Fr**0.56) )**1.25 # rearanged wright and parker (asce 110 eq 2-177)
    #     tau_b = tau_star * (self.rho_w*self.R*self.g*self.D50) 
    #     u_star = np.sqrt(tau_b / self.rho_w)
    #     Cz = V/u_star 
    #     entrainment_inputs = {'u star sk': u_star_sk, 'Fr': Fr, 'y sk':y_sk_solved}
    #     return Cz, entrainment_inputs
    
    def Cz_no_par(self,y,V,Fr):
        '''Take in stage and velocity as known during GVF steps. Calculate dinmensionless Chezy coefficient
        assuming no bedform drag.
        
        '''

        Cz = ManningStrickler(y,self.D50,3,alpha_r=8.32)/1
        u_star = V/Cz 
        entrainment_inputs = {'u star sk': u_star, 'Fr': Fr}
        return Cz, entrainment_inputs
    
    def E_s(self,entrainment_inputs):
        '''Return the concentration (at .05H) and the Rouse number using the reccomended equations from 
        de Leeuw et al. (2020).

        entrainment inputs is a dictionary with the following keys:
            u_star_sk: shear velocity from skin friction alone 
            v_s: settling velocity
        Fr: Froude number
        '''

        u_star_sk = entrainment_inputs['u star sk']
        Fr = entrainment_inputs['Fr']
        E_s =  4.74e-4 * (u_star_sk/self.v_s)**1.77 * Fr**1.18 
        P = (u_star_sk/self.v_s)**(-0.45)
        return E_s, P

    
    def C_to_Cb(self,C,P):
        '''For a given average volumetric concentration, calculate the concentration at the reference height 
        of .05 y (flow depth).

        P: Rouse number    
        y: flow depth
        '''

        J1 = RouseFits(P,self.a_ref)[0]

        Cb = C/J1 # ASCE110 2-215
        return Cb
    
    def v_s(self):
        '''Calculate settling velocity as in DeLeeuw et al. (2020), who used the relation of Fergusion and Church (2004). 

        C_1 and C_2 are hardcoded here to the same values used in the paper.
        '''
        C_1 = 18. 
        C_2 = 1. 
        
        v_s = (self.R*self.g*self.D50**2.) / ( C_1*self.nu + (0.75*C_2*self.R*self.g*self.D50**3)**0.55 )
        return v_s
    

    def normalFlowWide(self,y,S):
        '''Calculate the normal flow hydraulics and equilibrium sediment transport for a wide channel. 
        Solution approach: use the Einstein partition with the empirical relationship between skin friction
        and total friction from Wright and Parker (2004b). Sequence:
        1) Guess a depth of flow from the skin friction component
        2) Calculate the bed shear stress (normal flow condition) and shields number for the skin drag component.
        3) Use the empirical relation between skin friction shields stress and total shields stress
        4) Calculate the new shear stress. 
        5) Calculate normal flow conditions using that shear stress.
        6) Iterate on above to get a normal flow solution that matches the target depth.
        
        y 
        '''
        
        # use iterative solver to get the guess of the skin drag part of flow depth
        y_sk_guess = 0.5*y
        def f(y_sk):
            tau_b_sk = self.rho_w*self.g*y_sk*S
            tau_star_sk = tau_b_sk / (self.rho_w*self.R*self.g*self.D50)

            Cz_sk = ManningStrickler(y_sk,self.D50,3) # FIXME - FIXED to be the height corresponding to skin drag
            u_star_sk = np.sqrt(tau_b_sk/self.rho_w)
            U = Cz_sk*u_star_sk
            Fr = U / np.sqrt(self.g*y) # calculate the Froude number as if this iteration is turning out to give the correct flow depth

            tau_star = ( (tau_star_sk-.05)/(0.7*Fr**0.56) )**1.25 # rearanged wright and parker (asce 110 eq 2-177) 

            tau_b = tau_star * (self.rho_w*self.R*self.g*self.D50)

            y_guess = tau_b / (self.rho_w*self.g*S)

            return y_guess-y 
        
        y_sk_solved = fsolve(f,y_sk_guess)

        # with this known, calculate discharge, velocity, and equilibrium sediment transport.
        tau_b_sk = self.rho_w*self.g*y_sk_solved*S
        tau_star_sk = tau_b_sk / (self.rho_w*self.R*self.g*self.D50)

        Cz_sk = ManningStrickler(y,self.D50,3) # FIXME this should be using the assumed depth corresponding to skin drag
        u_star_sk = np.sqrt(tau_b_sk/self.rho_w)
        U = Cz_sk*u_star_sk
        Fr = U / np.sqrt(self.g*y)
        q = U*y # discharge per width

        entrainment_inputs = {'u star sk': u_star_sk, 'Fr': Fr}

        c_b, P = self.E_s(entrainment_inputs)
        c = c_b*RouseFits(P,self.a_ref)[0]

        results_out = {'y_sk': y_sk_solved, 'U': U, 'Fr': Fr, 'q': q, 'c': c, 'c_b': c_b, 'P': P}

        return results_out 

# endregion

# -------------------------------------------------------------------------------------------------
# region HYDRAULIC GEOMETRY METHODS
class TrapezoidChannel:
    '''Make a class that stores the channel geometry and returns cross section information at requested locations. 
    Idea is that it stores only the geometry itself and calculations based on that geometry. Does not store the water
    levels but provides normal and critical depths, Fr, etc as function of given flows.
    Stores the following variables
        x: x points (must start at zero)
        Y_B: base of channel in some fixed coordinate system 
        b: base of trapezoid. Can be given as array length as x or single value. 
        m: slope of trapezoid walls. Can be given as array length as x or single value. 
        n: mannings n. Can be given as array length as x or single value.  
        g: grav accel. 
        Kn: modifier for mannnings n. Only used if using SI units for some reason.
        nx: number of x points 
        S: slope (as positive) calculated with central differences from x
    '''

    def __init__(self,x,Y_B,b,m,n,g=9.81,Kn=1.0):
        self.x = x 
        self.Y_B = Y_B.copy()
        
        if np.size(np.array(b)) == 1:
            self.b = b * np.ones_like(x)
        else:
            self.b = b

        if np.size(np.array(m)) == 1:
            self.m = m * np.ones_like(x)
        else:
            self.m = m

        if np.size(np.array(n)) == 1:
            self.n = n * np.ones_like(x)
        else:
            self.n = n

        self.g = g
        self.Kn = Kn

        self.nx = np.size(x)
        self.S = -np.gradient(self.Y_B,self.x)
        #self.S_reverse = np.flip(self.S) 

    def update_bed(self,dY):
        '''Apply a geometry increment to the bed
        
        '''
        self.Y_B+=dY
        self.S = -np.gradient(self.Y_B,self.x)

    def A(self,y,ii):
        '''Area
        
        '''
        return y*(self.b[ii]+self.m[ii]*y)
    
    def y_of_A(self,A,ii): 
        '''Depth from area
        
        '''
        return 0.5*( (-self.b[ii]/self.m[ii]) + np.sqrt((self.b[ii]/self.m[ii])**2 - 4*(A/self.m[ii])) )
    
    def P(self,y,ii):
        '''Perimeter
        
        '''
        return self.b[ii]+2.*y*np.sqrt(1+self.m[ii]**2.)
    
    def R(self,y,ii):
        '''Hydraulic radius
        
        '''
        return self.A(y,ii) / self.P(y,ii)
    
    def Fr(self,y,V,ii,alpha=1.):
        '''Froude number
        
        '''
        B = self.b[ii] + 2*self.m[ii]*y
        D = self.A(y,ii)/B # hyrdraulic depth
        return V / ( (self.g*D/alpha)**0.5 )
    
    def y_c(self,Q,ii,alpha=1.):
        '''Critical depth at point'''
        Z = ( Q*self.m[ii]**1.5 ) / ( (self.g/alpha)**0.5 * self.b[ii]**2.5 )
        def func(yp):
            return ( (yp*(1+yp))**1.5 ) / ( (1+2*yp)**0.5 ) - Z 
    
        solve_out = fsolve(func, 1.)
        #print('crit depth solver:', solve_out)
        yp_solved = solve_out[0]
        yc = yp_solved*self.b[ii]/self.m[ii]
        return yc
    
    def y_0(self,Q,ii):
        '''Normal depth
        
        '''
        RHS_func = self.n[ii]*Q / (self.Kn * self.S[ii]**0.5 * self.b[ii]**(8/3))
        def func(yp):
            return (yp*(1+self.m[ii]*yp))**(5/3) / (1+2*yp*(1+self.m[ii]**2.)**0.5)**(2/3) - RHS_func
        yp_solved = fsolve(func, 0.5)[0] 
        y0 = yp_solved*self.b[ii]
        return y0 
    
    def y_c_prof(self,Q,alpha=1.):
        '''Calculate critical depth for entire profile
        
        '''
        y_c_of_x = np.zeros_like(self.x)
        for aa in np.arange(0,self.nx):
            y_c_of_x[aa] = self.y_c(Q,aa,alpha=alpha)
        return y_c_of_x
    
    def y_0_prof(self,Q):
        '''Calculate normal depth for entire profile
        
        '''
        y_0_of_x = np.zeros_like(self.x)
        for aa in np.arange(0,self.nx):
            y_0_of_x[aa] = self.y_0(Q,aa)
        return y_0_of_x
    
    def plot_geometry(self,fig=None,ax=None):
        if ax is None:
            fig, ax = plt.subplots(3,1,figsize=(8,6),layout='constrained')
            ax[0].plot(self.x,self.Y_B)
            ax[0].set_xlabel('x (m)')
            ax[0].set_ylabel('Y_B (m)')
            ax[0].set_title('Channel bed elevation')
            ax[1].plot(self.x,self.b)
            ax[1].set_xlabel('x (m)')
            ax[1].set_ylabel('B (m)')
            ax[1].set_title('Channel width')
            ax[2].plot(self.x,self.S)
            ax[2].set_xlabel('x (m)')
            ax[2].set_ylabel('S (m/m)')
            ax[2].set_title('Channel slope')
        else:
            ax[0].plot(self.x,self.Y_B)
            ax[1].plot(self.x,self.Y_B)
            ax[2].plot(self.x,self.S)
        return fig, ax


# endregion

# -------------------------------------------------------------------------------------------------
# region HYDRAULIC SOLVERS

def SubcriticalGVF(chan,Q,y_out,transport=None):  
    '''Solve the gradually varied flow problem for subcritical condtions. Solve ODE from 
    downstream to upstream with given channel properties, dischrage, and exit height 
    (relative to the channel base).

    chan: an instance of a channel geometry class. Currently the only channel class is 
    the trapezoid channel, but a class for some other geometry that has the same methods
    should be plug and play.
    Q: discharge (m^3/s). Assumed constant.
    y_out: height above channel base at exit (m). Function detects if this value is 
    less than critical depth and adjusts exit to slightly over critical depth accordingly.
    transport: overrides the Manning's n in the channel instance with a 
    ''' 

    y_c = chan.y_c(Q,-1) 
    #y_0_exit = chan.y_0(Q,-1)

    if y_c < y_out:
        y_exit = y_out 
        exit_type = 'subcritical'
    else:
        y_exit = y_c*1.01
        exit_type = 'critical'

    if transport is None:
        def f(t,y): # t is the x because we're solving a BVP as an ODE
            
            # flip the x values and get the index from the nearest match. 
            # for this to work requires the x points in channel to have
            # equal spacing and to start from zero.
            ii = np.argmin(np.abs(np.flip(chan.x)-t)) 
                                                    
            A = chan.A(y,ii)
            V = Q/A
            #E = SpecificEnergy(y,V)
            R = chan.R(y,ii)
            SE = EnergySlope(chan.n[ii],V,R)
            F = chan.Fr(y,V,ii)
            if F >= 1:
                raise ValueError('Fr>1, SubcriticalGVG function only handles subcritical GVF')
            S0 = chan.S[ii]
            return -(S0-SE)/(1-F**2)
        
    else:
        def f(t,y): # t is the x because we're solving a BVP as an ODE
            
            # flip the x values and get the index from the nearest match. 
            # for this to work requires the x points in channel to have
            # equal spacing and to start from zero.
            ii = np.argmin(np.abs(np.flip(chan.x)-t)) 
                                                    
            A = chan.A(y,ii)
            V = Q/A
            #E = SpecificEnergy(y,V)
            R = chan.R(y,ii)
            F = chan.Fr(y,V,ii)
            #print(F)
            #if F >= 1:
            #    raise ValueError('Fr>1, solver only handles subcritical')
            #print('ahh',y)
            #Cz = transport.Cz(y,V,F)[0]
            Cz = transport.Cz(y,V,F)[0]
            n = ManningFromChezy(R,Cz) # FIXME havent thought too much about impact of cross section.
            SE = EnergySlope(n,V,R)
            S0 = chan.S[ii]
            return -(S0-SE)/(1-F**2)


    solve_out = solve_ivp(f, [chan.x[0],chan.x[-1]], [y_exit], t_eval=chan.x)

    y = np.flip(solve_out.y[0])

    return y

def ReservoirHeight(y_in,Q,A_in,V_res=0.0,g=9.81):
    '''Calculate the reservoir height required to drive an inflow with some height and discharge. Conceptually
    the gradually varied flow problem will have provided a height and discharge at the start of the channel 
    downstream of the reservoir. For a still reservoir, the height must exceed the channel height according to 
    the energy equation. V_res allows for some velocity in the channel direction adding energy.
    
    y_in: the height of flow in the channel entrance (over the base) 
    Q: discharge in the channel
    A_in: area of flow in channel entrance
    V_res: velocity in reservoir in channel direction.
    '''
    # FIXME having y_in work is only the case for a rectangular or trapezoidal channel otherwise would need centroid??
    V_in = Q/A_in
    y_res = y_in + V_in**2/(2*9.81) - V_res**2 / (2*9.81)
    return y_res


def ReservoirProblemSubcritical(chan,y_res_in,y_out,V_res=0.,transport=None,Q_guess=None):
    '''Solve gradually varied flow between two reservoirs. If the slope is constant for long enough exiting the 
    upper reservoir, normal flow will occur and it's possible to solve for the normal flow that respects the energy 
    equation for the reservoir height turning into height and velocity. For arbitrary channel geometries, however, 
    this is not possible and this function handles it. Uses the shooting method namely finding the gradually varied
    flow solution that satisfies the energy equation for the reservoir flow into the start of the channel.

    chan: an instance of a channel geometry class. Currently the only channel class is 
    the trapezoid channel, but a class for some other geometry that has the same methods
    should be plug and play.
    y_res_in: height of the reservoir above the channel inlet base
    y_out: downstream height above base boundary condtion
    V_res: velocity in reservoir towards the channel.
    Q_guess: the discharge estimate to use. NOTE: if modifying the channel or flow BC incrementally, guessing 
        the last solution can help a lot with speed and even finding a solution.
    '''

    # make an initial guess assuming at normal flow
    nx_avg_to = int(chan.x[-1]/4)
    avg_inds = np.arange(0,nx_avg_to)
    S_in_avg = np.mean(chan.S[0:nx_avg_to]) # use the average slope of the first 1/4 of the channel
    n_in_avg = np.mean(chan.n[avg_inds])
    y_in_guess = 0.9*y_res_in # assume that y_in at will be close to y_res_in because subcritical 
    A_in_guess = np.mean(chan.A(y_in_guess,avg_inds))
    R_in_guess = np.mean(chan.R(y_in_guess,avg_inds))
    
    if Q_guess is None:
        Q_guess = (chan.Kn/n_in_avg) * A_in_guess * R_in_guess**(2/3) * S_in_avg**0.5 
    #print(Q_guess,n_in_avg,chan.Kn,A_in_guess,R_in_guess,S_in_avg)

    def func(Q_g):
        #print('Test GVF')
        y_g_all = SubcriticalGVF(chan,Q_g,y_out,transport=transport)
        y_g = y_g_all[0] 
        A_g = chan.A(y_g,0)
        y_res_g = ReservoirHeight(y_g,Q_g,A_g,V_res=V_res) 
        #print(np.shape(y_res_g))
        #print(np.shape(y_res_in))
        return y_res_g - y_res_in 
    
    Q_match = fsolve(func, Q_guess, )[0] 

    return Q_match

# endregion

# -------------------------------------------------------------------------------------------------
# region SEDIMENT TRANSPORT SOLVERS

class SedimentResults:
    def __init__(self,Q_t,c,c_b,E_s,eta_dot):
        self.Q_t = Q_t 
        self.c = c 
        self.c_b = c_b 
        self.E_s = E_s
        self.eta_dot = eta_dot 

def QuasiSteadyTransport(chan,transport,y,Q,c_in) -> SedimentResults:
    '''Solve the sediment continuity equation to get erosion and deposition rates across the domain. Ref ASCE 110 eq 3-135
    and 3-136 for 1d formulation. Modified slightly to consider trapezoid geometry. Also, return the bed elevation change 
    rate (assuming constant active layer thickness). Ref ASCE110 eq 3-136. Note that we could use the divergence of Q_t,
    but that would pick up a discretization error that we can avoid by using the differential equation to calculate the
    change rate rather than the other way around.
    
    Solving the equation: FIXME
    where v_s is settling velocity, c_b is the concentration (volume) at a ref elevation from the bed, delta_t is the time
    required for flow to pass through the cell, B is the width of the bed, delta_x is the element length, E_s, is the 
    entrainment rate from a suspended transport entrainment function. c_b is related to c_bar (depth average concentration)
    via the Rouse profile using a fit relations.

    chan: channel instance
    transport: transport instance
    y: depth as func of x
    Q: water discharge (constant)
    c_in: concentration of sediment coming in

    Returns dictionary with:
    Q_t: sediment volumetric flow rate
    c: sediment concentration average
    c_b: sediment concentration at near bed
    eta_dot: bed aggredation / degradation rate (agg is positive)

    NOTE: would need to adjust hardcoded .05 H usage if a different transport relation were to be implemented
    '''

    A_of_x = chan.A(y,np.arange(0,chan.nx))  
    V_of_x = Q / A_of_x 

    Fr_of_x = chan.Fr(y,V_of_x,np.arange(0,chan.nx))
    
    Cz_of_x, entrain_in = transport.Cz(y,V_of_x,Fr_of_x)
    E_s_of_x, P_of_x =  transport.E_s(entrain_in)

    dx = chan.x[1] - chan.x[0]
    v_s = transport.v_s

    def f(x,Q_t):
        V = np.interp(x,chan.x,V_of_x)
        #t_pass = dx / V 

        B = np.interp(x,chan.x,chan.b)
        P = np.interp(x,chan.x,P_of_x) # this is probably not strictly correct. Using the empirical rouse exponent
                                       # from De Leeuw, which likely corresponds to stratification at equilibtrium?
        #A = np.interp(x,chan.x,A_of_x)
        J1 = RouseFits(P,transport.a_ref)[0]
        c = Q_t/Q
        c_b = c/J1 # J1 provides the ratio of concentration at the .05H level to avg concentration.

        E_s = np.interp(x,chan.x,E_s_of_x)

        #return -v_s*c_b*(t_pass*B*dx) + v_s*E_s*(t_pass*B*dx)
        #return -v_s*c_b*B/A + v_s*E_s*B/A
        #return B*v_s*(c_b-E_s)
        return B*v_s*(E_s-c_b)

    # FIXME this IVP is apparently pretty stiff. LSODA method seems to do best. 
    solve_out = solve_ivp(f, [chan.x[0],chan.x[-1]], [c_in*Q], t_eval=chan.x, method="LSODA")
    Q_t_soln = solve_out.y[0]

    # back calculate average concentration and near-bed concentration
    c_soln = Q_t_soln / Q 
    c_b_soln = c_soln / RouseFits(P_of_x,transport.a_ref)[0]

    # back calculate the bed change rate
    eta_dot = -( v_s*(E_s_of_x-c_b_soln) ) / (1-transport.lamb) 
    
    sed_out = SedimentResults(Q_t_soln,c_soln,c_b_soln,E_s_of_x,eta_dot)
    return sed_out
    #return {'Q_t': Q_t_soln, 'c': c_soln, 'c_b': c_b_soln, 'eta_dot': eta_dot}


# endregion

# -------------------------------------------------------------------------------------------------
# region PLOTTING FUNCTIONS 

def PlotHydraulics(chan,y,Q,plot_critical=True,plot_normal=True,ax=None,color=None,label=None):
    '''Function to show long profile
    
    
    chan: an instance of a channel geometry class. Currently the only channel class is 
    the trapezoid channel, but a class for some other geometry that has the same methods
    should be plug and play.
    y: flow height above base profile on same points as chan.x 
    Q: discharge (constant over channel distance)
    '''
    

    # calc profiles in absolute coordinates
    Y = chan.Y_B + y  # water surface 
    y_c = chan.y_c_prof(Q) # critical depth
    Y_c = chan.Y_B + y_c # critcal depth surface
    y_0 = chan.y_0_prof(Q) # normal depth 
    Y_0 = chan.Y_B + y_0 # normal depth surface

    A_of_x = chan.A(y,np.arange(0,chan.nx)) 
    V_of_x = Q / A_of_x 
    #print(V_of_x[-1])
    Fr_of_x = chan.Fr(y,V_of_x,np.arange(0,chan.nx))

    # plot
    return_var = None
    if ax is None:
        fig, ax = plt.subplots(2,2,figsize=(12,6),layout='constrained')
        return_var = (fig, ax)

        ax[0,0].plot(chan.x,y,c=color)

        ax[0,1].plot(chan.x,chan.Y_B,c=color,label='Bed')
        ax[0,1].plot(chan.x,Y,'b',label='Surf.')
        if plot_critical:
            ax[0,1].plot(chan.x,Y_c,'r--',label='Crit.')
        if plot_normal:
            ax[0,1].plot(chan.x,Y_0,'k--',label='Norm.')

        ax[1,0].plot(chan.x,V_of_x,c=color)

        ax[1,1].plot(chan.x,Fr_of_x,c=color,label=label)

        # clean up 
        ax[0,0].set_ylabel('y (m)')
        ax[0,0].set_title('Depth')

        ax[0,1].set_ylabel('Y (m)')
        ax[0,1].legend()
        ax[0,1].set_title('Long Profile')
        if plot_critical:
            ax[0,1].plot(chan.x,Y_c,'r--',label='Crit.')
        if plot_normal:
            ax[0,1].plot(chan.x,Y_0,'g--',label='Norm.')

        ax[1,0].set_xlabel('x (m)')
        ax[1,0].set_ylabel('V (m/s)')
        ax[1,0].set_title('Velocity')
        
        ax[1,1].set_xlabel('x (m)')
        ax[1,1].set_ylabel('Fr')
        ax[1,1].set_ylim([0,1])
        ax[1,1].set_title('Froude Number')

    else:
        ax[0,0].plot(chan.x,y,c=color)

        ax[0,1].plot(chan.x,chan.Y_B,c=color,label='Bed')
        ax[0,1].plot(chan.x,Y,'b',label='Surf.')
        if plot_critical:
            ax[0,1].plot(chan.x,Y_c,'r--') #,label='Crit.')
        if plot_normal:
            ax[0,1].plot(chan.x,Y_0,'g--') #,label='Norm.')

        ax[1,0].plot(chan.x,V_of_x,c=color)
        #ax[1,0].set_ylim([0,None])

        ax[1,1].plot(chan.x,Fr_of_x,c=color,label=label)

    ax[1,1].legend(ncols=2)
    return return_var

def PlotSediment(chan,trans,y,Q,SedimentResults,ax=None,color='k',label=None):

    A_of_x = chan.A(y,np.arange(0,chan.nx)) 
    V_of_x = Q / A_of_x 
    Fr_of_x = chan.Fr(y,V_of_x,np.arange(0,chan.nx))
    Cz = trans.Cz(y,V_of_x,Fr_of_x)[0]

    Y = chan.Y_B + y

    return_var = None
    if ax is None:
        fig, ax = plt.subplots(2,2,figsize=(12,6),layout='constrained')
        return_var = (fig, ax)

        ax[0,0].plot(chan.x,SedimentResults.c_b,c=color)
        ax[0,0].plot(chan.x,SedimentResults.E_s,'--',c=color)
        ax[0,0].set_ylabel('concentration (volumetric)')
        ax[0,0].legend(['c_b','E_s'])
        ax[0,0].set_title('%.2fH entrainment and concentration' % trans.a_ref)

        ax[0,1].plot(chan.x,SedimentResults.c,c=color,label=label)
        ax[0,1].set_ylabel('concentration (volumetric)')
        ax[0,1].set_title('Average concentration')

        ax[1,0].plot(chan.x,SedimentResults.eta_dot,c=color)
        ax[1,0].plot([chan.x[0],chan.x[-1]], [0,0], 'r--')
        ax[1,0].set_xlabel('x (m)')
        ax[1,0].set_ylabel(r'$\eta$ (m/s)')
        ax[1,0].set_title('Bed change rate')

        ax[1,1].plot(chan.x,chan.Y_B,'k',label='Bed',c=color)
        ax[1,1].plot(chan.x,Y,'b',label='Surf.')
        ax[1,1].set_xlabel('x (m)')
        ax[1,1].set_ylabel('Y')
        ax[1,1].set_title('Bed and surface')

        # ax[1,1].plot(chan.x,Cz)
        # ax[1,1].set_xlabel('x (m)')
        # ax[1,1].set_ylabel('Cz')
        # ax[1,1].set_title('Dimensionless Chezy number')

    else:
        return_var = None
        ax[0,0].plot(chan.x,SedimentResults.c_b,c=color)
        ax[0,0].plot(chan.x,SedimentResults.E_s,'--',c=color)

        ax[0,1].plot(chan.x,SedimentResults.c,c=color,label=label)

        ax[1,0].plot(chan.x,SedimentResults.eta_dot,c=color)
        ax[1,0].plot([chan.x[0],chan.x[-1]], [0,0], 'r--')

        ax[1,1].plot(chan.x,chan.Y_B,c=color,label='Bed')
        ax[1,1].plot(chan.x,Y,'b',label='Surf.')

        # ax[1,1].plot(chan.x,Cz)
        # ax[1,1].set_xlabel('x (m)')
        # ax[1,1].set_ylabel('Cz')
        # ax[1,1].set_title('Dimensionless Chezy number')
    

    ax[0,1].legend(ncols=2)
    return return_var

# endregion


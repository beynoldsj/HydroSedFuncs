import pandas as pd
import numpy as np 
import scipy as sp
import seaborn as sns
from lmoments3 import distr
import matplotlib.pyplot as plt

'''
This module contains functions for hydrology statistics.
Ben Reynolds, 2026
'''

# -------------------------------------------------------------------------------------------------
# region Interpolation functions for log and semilogy in base 10 and base 2

def log10_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def semilog10y_interp1d(xx, yy, kind='linear'):
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(xx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(zz))
    return log_interp

def semilog10x_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    lin_interp = sp.interpolate.interp1d(logx, yy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def log2_interp1d(xx, yy, kind='linear'):
    logx = np.log2(xx)
    logy = np.log2(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(2.0, lin_interp(np.log2(zz)))
    return log_interp

def semilog2y_interp1d(xx, yy, kind='linear'):
    logy = np.log2(yy)
    lin_interp = sp.interpolate.interp1d(xx, logy, kind=kind)
    log_interp = lambda zz: np.power(2.0, lin_interp(zz))
    return log_interp

def semilog2x_interp1d(xx, yy, kind='linear'):
    logx = np.log2(xx)
    lin_interp = sp.interpolate.interp1d(logx, yy, kind=kind)
    log_interp = lambda zz: np.power(2.0, lin_interp(np.log2(zz)))
    return log_interp



# endregion

# -------------------------------------------------------------------------------------------------
# region Flood frequency statistics



def LogPearson3Return(peak_discharges,boot_meth='param',n_boot=1000,chan_form_return=1.5,make_plots=True):
    '''
    Calculate the discharge versus return period given peak annual discharges as input. Use the Log Pearson 3 
    distribution, which works well for extreme value distributions (ref USGS Bulletin 17C). This method does
    not fully follow Bulletin 17C which uses information from other regional rivers to adjust skew. This method
    is just a fit to the given data. It uses the L-moment method to fit, which matches paramaters of the given 
    distribution, rather than a maximum likelihood estimator (MLE) approach, because L-moment is more robust for
    sparse data. 
    
    peak_discharges: dataframe with a column 'peak_va_cms' which should contain peak annual discharges
    in cubic meters per second (CMS).
    boot_meth: defaults to 'param' can be 'non-param'. Parametric bootstrapping fits the distribution to all the
    data, and then produces synthetic data and fits distributions to the synthetic data to estimate uncertainty of
    the fit. It's appropriate when data are limited but the expected distribution is known. Non-parametric would 
    generally require more data.

    '''


    peak_discharges['rank'] = peak_discharges['peak_va_cms'].rank()
    N = np.sum(~np.isnan(peak_discharges['peak_va_cms']))
    peak_discharges['exc_prob'] = (N - peak_discharges['rank'] + 1)/(N+1)
    peak_discharges['return'] = 1. / peak_discharges['exc_prob']

    # FIT MODEL 
    x_r80 = np.linspace(np.round(0.5*peak_discharges['peak_va_cms'].min(),-3),np.round(1.2*peak_discharges['peak_va_cms'].max(),-3),1000) 
    
    params_pe3 = distr.pe3.lmom_fit(np.log10(peak_discharges['peak_va_cms']))
    prob_r80_p = distr.pe3.pdf(np.log10(x_r80), params_pe3["skew"], params_pe3['loc'], params_pe3['scale'])
    cum_prob_r80_p = distr.pe3.cdf(np.log10(x_r80), params_pe3["skew"], params_pe3['loc'], params_pe3['scale'])

    #print(params_pe3)

    # BOOTSTRAP FOR UNCERT
    # initalize list to store parameters from samples
    params = []
    
    # generate n_boot samples by resampling generated functions
    for i in range(n_boot):
        # THIS METHOD IS THE PARAMETRIC BOOTSTRAP
        if boot_meth == 'param':
            #boot_data = sp.stats.genextreme.rvs(shape,loc=loc,scale=scale,size=N)
            boot_data = distr.pe3.rvs(params_pe3["skew"], params_pe3['loc'], params_pe3['scale'], size=N)
    
        
        # THE LINE BELOW IS FOR NON-PARAMETRIC BOOTSTRAP, WHICH IS NOT LEGIT FOR SMALL SAMPLES AND EXTRAPOLATION. 
        # AS WRITTEN IT MAY BE PULLING TOO MANY SAMPLES ? SEE GILLELAND (2020)
        elif boot_meth == 'non-param':
            boot_data = np.random.choice(peak_discharges['peak_va_cms'], size=peak_discharges['peak_va_cms'].count(), replace=True)
    
        else:
            print('Wahhh')
            break
    
        #params.append(sp.stats.genextreme.fit(boot_data,0))
        params.append(distr.pe3.lmom_fit(boot_data))
    
    '''
    # print the estimate of the mean of each parameter and it's confidence intervals
    print(
        "Mean estimate: ",
        np.mean(np.array(params), axis=0),
        " and 95% confidence intervals: ",
        np.quantile(np.array(params), [0.025, 0.975], axis=0),
    )
    '''
    
    # generate years vector
    years = np.arange(1, 100, 0.01)
    probs = 1/years
    
    # intialize list for return levels
    levels = []
    
    # calculate return levels for each of the n_boot samples
    for i in range(n_boot):
        #levels.append(sp.stats.genextreme.ppf((1.-1./years), *params[i]))
        #levels.append(10**distr.pe3.ppf((1.-1./years), *params[i]))
        levels.append(10**distr.pe3.ppf((1.-1./years), params[i]["skew"], params[i]["loc"], params[i]["scale"]))
        
    levels = np.array(levels)

    #sum_df.at[nn,'Q_chan'] = 10**distr.pe3.ppf((1.-1./chan_form_return), params_pe3["skew"], params_pe3["loc"], params_pe3["scale"])
    #sum_df.at[nn,'chan_int'] = chan_form_return
    flow_est = 10**distr.pe3.ppf((1.-1./chan_form_return), params_pe3["skew"], params_pe3["loc"], params_pe3["scale"])
    flow_est_hi = np.interp(np.log10(chan_form_return),np.log10(years),np.quantile(levels, [0.975], axis=0)[0,:])
    flow_est_lo = np.interp(np.log10(chan_form_return),np.log10(years),np.quantile(levels, [0.025], axis=0)[0,:])
    flow_est_hi_error = flow_est_hi-flow_est
    flow_est_lo_error = flow_est-flow_est_lo

    # MAKE PLOTS
    if make_plots:
        # setup plots
        fig, ax = plt.subplots(1,2,figsize=(8,3), layout='constrained') 
        
        # find empirical return levels
        ax[0].semilogx(peak_discharges['return'], peak_discharges['peak_va_cms'],'x',label=('observed (N=%d)' % N))
        
        # plot best fit 
        ax[0].semilogx(1/(1-cum_prob_r80_p),x_r80,'g:',label='best fit')

        #ax[0].semilogx(1/(1-chan_form_return),sum_df.at[nn,'Q_chan'],'ro')
        
        # plot return mean levels
        ax[0].semilogx(years, levels.mean(axis=0), label='mean')
        
        # plot confidence intervals
        ax[0].semilogx(years, np.quantile(levels, [0.025], axis=0).T, "k--", label='95% CI')
        ax[0].semilogx(years, np.quantile(levels, [0.975], axis=0).T, "k--")

        ax[0].errorbar(chan_form_return,flow_est,yerr=[[flow_est_lo_error],[flow_est_hi_error]],
                    color='red',fmt='o',label='1.5 year est.',markersize=4)
        
        ax[0].set_xlabel('Return period (yr)')
        ax[0].set_ylabel('Discharge (cms)')
        ax[0].set_title('Flow return periods')
        #ax[0].legend([('obs. (N=%d)' % N),'best fit',('%.1fyr' % chan_form_return),'mean','5/95 CI'])
        #ax[0].legend([('obs. (N=%d)' % N),'best fit','mean','95% CI'])
        ax[0].legend(fontsize=8)
        ax[0].set_xlim([0.9,101])


        ax[1].plot(peak_discharges.index, peak_discharges['peak_va_cms'],'x')
        ax[1].set_ylabel('Discharge (cms)')
        ax[1].grid('both')
        ax[1].set_title('Discharge through time')
        ax[1].tick_params("x", rotation=45)
        fig_ax = [fig, ax]

    else:
        fig_ax = [None, None]

    return flow_est, flow_est_hi, flow_est_lo, params_pe3, fig_ax





# endregion
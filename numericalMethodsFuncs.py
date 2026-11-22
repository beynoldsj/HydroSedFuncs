
import numpy as np


def SplineInterpolateEstimate(func, x_guess, spl_pts=np.array([-.3,-.2,-.1,0,.1,.2,.3])):
    '''
    Takes in a function and x_guess. Makes x_pts that are spl_pts times x_guess adjusted from x_guess
    Does spline fit to find value of x that makes value of y (the function output) zero.
    '''

    x_pts = x_guess + x_guess*spl_pts 

    y_pts = np.zeros_like(x_pts)
    y_pts[:] = np.nan
    ii = 0
    for x_pt in x_pts:
        y_pts[ii] = func(x_pt)
        ii += 1

    poly_coefs = np.polyfit(y_pts,x_pts,2)
    x_solved = np.polyval(poly_coefs,0.0)
    if x_solved < np.min(x_pts) or x_solved > np.max(x_pts):
        raise ValueError('spline_interp_estimate: spline pts did not cover range.')

    return x_solved



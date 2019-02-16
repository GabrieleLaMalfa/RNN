# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:05:10 2018

@author: Emanuele

Test if a dataset follows a particular distribution.
This snippet is taken from https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python,
 modified and adapted to give as result the top-n distributions and their respective errors.
"""

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

"""
 Model data by finding best fit distribution to data.
"""
def best_fit_distribution(data, bins=200, ax=None, top_n=3):
    
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    # Best holders
    best_distribution = DISTRIBUTIONS[:top_n]
    best_params = [(0.0, 1.0) for _ in range(top_n)]
    best_sse = [np.inf for _ in range(top_n)]

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    
                except Exception:
                    pass
                                  
                # identify if this distribution is in the top-n
                for i in range(top_n):
                    
                    if best_sse[i] > sse > 0:
                        
                        best_distribution[i] = distribution
                        best_params[i] = params
                        best_sse[i] = sse
                        break

        except Exception:
            pass
        
    # extract name's fomr top-n distributions
    best_distribution = [b.name for b in best_distribution]

    return (best_distribution, best_params, best_sse)


"""
 Generate distributions's Probability Distribution Function.
"""
def make_pdf(dist, params, size=10000):

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


"""
 Plot the best distribution on data.
"""
def calculate_best_fit_distribution(data):

    data = pd.Series(data)
    
    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5)
    # Save plot limits
    dataYLim = ax.get_ylim()
    
    # Find best fit distribution
    best_fit = best_fit_distribution(data, 200, ax)
    best_fit_name, best_fit_params = best_fit[0,0], best_fit[0,1]
    best_dist = getattr(st, best_fit_name)
    
    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title('Data distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    
    # Make PDF with best params 
    pdf = make_pdf(best_dist, best_fit_params)
    
    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)
    
    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)
    
    ax.set_title('Data distribution \n' + dist_str)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
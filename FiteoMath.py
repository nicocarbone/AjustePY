#!/usr/bin/env python2
# import argparse
# -*- coding: utf-8 -*-

"""
Created on Wed Feb  5 10:01:05 2014

@author: Nicolas Abel Carbone / ncarbone@exa.unicen.edu.ar

Mathematical functions
"""

# imports
import math
import numpy as np

# global variable declarations with default values
n_ref = 1.4
v = 299.792458000 / n_ref
if n_ref > 1:
    A = 504.332889 - 2641.00214 * n_ref + 5923.699064 * n_ref**2 - 7376.355814 * n_ref**3 + \
        5507.53041 * n_ref**4 - 2463.357945 * n_ref**5 + \
        610.956547 * n_ref**6 - 64.8047 * n_ref**7
if n_ref <= 1:
    A = 3.084635 - 6.531194 * n_ref + 8.357854 * \
        n_ref**2 - 5.082751 * n_ref**3 + 1.171382 * n_ref**4

sep = 52.  # Slab thickness
ro = 0.  # Separation between the source and the optical axis
data = []  # Array with experimental data
first_nonzero_data = 0  # First nonzero value in the data array
last_nonzero_data = 4095  # Last nonzero value in the data array
# data_norm = []  # Array with normalized experimental data
instru = []  # Array with response data
temp_data = []  # Array with temporal x axis
max_temp = 50.


def pre_calcs():
    """ Calculates A and v as a function of the refraction index,
    and the first and last non-zero values of the data

    TODO: more elegant way of doing this.
    """
    global v
    global A
    global first_nonzero_data
    global last_nonzero_data
    # baseline_sample = 50
    v = 299.792458000 / n_ref
    if n_ref > 1:
        A = 504.332889 - 2641.00214 * n_ref + 5923.699064 * n_ref**2 - 7376.355814 * n_ref**3 + \
            5507.53041 * n_ref**4 - 2463.357945 * n_ref**5 + \
            610.956547 * n_ref**6 - 64.8047 * n_ref**7
    if n_ref <= 1:
        A = 3.084635 - 6.531194 * n_ref + 8.357854 * \
            n_ref**2 - 5.082751 * n_ref**3 + 1.171382 * n_ref**4
    first_nonzero_data = np.nonzero(data)[0][0]
    # Calculate position of first nonzero value in data
    last_nonzero_data = np.nonzero(data)[0][-1]
    # Calculate position of last nonzero value in data
    # first_nonzero_instru = np.nonzero(instru)[0][0]
    # last_nonzero_instru = np.nonzero(instru)[0][-1]
    # baseline = sum(instru[first_nonzero_instru:
    # (first_nonzero_instru+baseline_sample)])/baseline_sample
    # instru[first_nonzero_instru:last_nonzero_instru] =
    # instru[first_nonzero_instru:last_nonzero_instru] - baseline


def smooth(x, window_len=10, window='hanning'):
    # Based on http://wiki.scipy.org/Cookbook/SignalSmooth
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    # print(len(s))

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


def term_sum(t, ups, ua, t0, m):
    """ Calculates the terms of the summatory of the slab model

    input:
        t: temporal X-axis
        ups: reduced scattering coefficient
        ua: absorption coefficient
        t0: temporal displacement
        m: number of the term to calculate
    """
    D = 1. / (3. * (ups + ua))
    ze = 2. * A * D
    z1 = sep * (1. - 2. * m) - 4. * m * ze - (1. / ups)
    z2 = sep * (1. - 2. * m) - (4. * m - 2.) * ze + (1. / ups)
    val1 = 4. * D * v * (t + t0)
    return z1 * math.e**(-z1**2. / val1) - z2 * math.e**(-z2**2. / val1)


def funcion_teo_ex_slab(t, ups, ua, t0):
    """ Calculates the thoretical slab model (Contini)

    input:
        t: temporal X-axis
        ups: reduced scattering coefficient
        ua: absorption coefficient
        t0: temporal displacement
    """
    dipolos = 10
    suma = 0
    for i in range(-dipolos / 2, dipolos / 2 + 1, 1):
        suma += term_sum(t, ups, ua, t0, i)
    D = (1. / (3. * (ups + ua)))
    val2 = 4. * math.pi * D * v
    pow1 = val2 ** 1.5
    pow2 = (t + t0) ** 2.5
    pow3 = ro ** 2
    val1 = 4. * D * v * (t + t0)
    val3 = -(ua * v * (t + t0)) - pow3 / val1
    pow4 = math.e ** val3
    return (pow4 / (2. * pow1 * pow2)) * suma


def funcion_teo_refl(t, ups, ua, t0):
    """ Calculates the thoretical semi-infinite reflectance model (Contini)

    input:
        t: temporal X-axis
        ups: reduced scattering coefficient
        ua: absorption coefficient
        t0: temporal displacement
    """
    z0 = 1. / ups
    D = 1. / (3. * (ups + ua))
    ze = 2. * A * D
    val1 = ua * v * (t + t0)
    val2 = (ro ** 2.) / (4. * D * v * (t + t0))
    val3 = 4. * math.pi * D * v
    val4 = 2. * ze + z0
    pow1 = math.e ** (-val1 - val2)
    pow2 = val3 ** 1.5
    pow3 = (t + t0) ** 2.5
    pow4 = z0 ** 2.
    pow5 = math.e ** ((-pow4) / (4. * D * v * (t + t0)))
    pow6 = val4 ** 2.
    pow7 = math.e ** ((-pow6) / (4. * D * v * (t + t0)))
    return ((-pow1) / (2. * pow2 * pow3)) * (((-z0) * pow5) - (val4 * pow7))


def funcion_fiteo_refl(t, ups, ua, t0, back):
    """ Calculates the convoluted function to fit - Semi-infinite model
    It creates an array using the theoretical model, convolutes it with the responce function
    and returns the value for any t value through lineal interpolation.

    input:
        t: temporal X-axis
        ups: reduced scattering coefficient
        ua: absorption coefficient
        t0: temporal displacement
        back: additive background level
    """
    array_teo = funcion_teo_refl(temp_data, ups, ua, t0)
    # Normalize the theoretical array by maximum
    #array_teo = array_teo / array_teo.max()  
    # Normalize the theoretical array by area
    array_teo = array_teo / np.trapz(array_teo)
    # Convolve with response function
    array_conv = np.convolve(instru, array_teo, 'full')
    # Truncate resultiing array to the size of the experimental data
    array_conv.resize(data.size)
    # Normalize the convoluted array by maximum
    #array_conv = array_conv / array_conv.max()
    # Normalize the theoretical array by area
    array_conv = array_conv / np.trapz(array_conv)
    # Set to zero the same positions as in the data array
    array_conv[:first_nonzero_data] = 0
    # Set to zero the same positions as in the data array
    array_conv[last_nonzero_data:] = 0
    # Return the final value for t, plus a background constant level
    return np.interp(t, temp_data, array_conv) + back


def funcion_fiteo_slab(t, ups, ua, t0, back):
    """ Calculates the convoluted function to fit - Semi-infinite model
    It creates an array using the theoretical model, convolutes it with the responce function
    and returns the value for any t value through lineal interpolation.

    input:
        t: temporal X-axis
        ups: reduced scattering coefficient
        ua: absorption coefficient
        t0: temporal displacement
        back: additive background level
    """
    array_teo = funcion_teo_ex_slab(temp_data, ups, ua, t0)
    # Normalize the theoretical array by maximum
    #array_teo = array_teo / array_teo.max()  
    # Normalize the theoretical array by area
    array_teo = array_teo / np.trapz(array_teo)
    # Convolve with response function
    array_conv = np.convolve(instru, array_teo, 'full')
    # Truncate resultiing array to the size of the experimental data
    array_conv.resize(data.size)
    # Normalize the convoluted array by maximum
    #array_conv = array_conv / array_conv.max()
    # Normalize the theoretical array by area
    array_conv = array_conv / np.trapz(array_conv)
    # Set to zero the same positions as in the data array
    array_conv[:first_nonzero_data] = 0
    # Set to zero the same positions as in the data array
    array_conv[last_nonzero_data:] = 0
    # Return the final value for t, plus a background constant level
    return np.interp(t, temp_data, array_conv) + back

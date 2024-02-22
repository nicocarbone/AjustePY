#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Feb  5 10:01:05 2014

@author: Nicolas Abel Carbone / ncarbone@exa.unicen.edu.ar

Mathematical functions
"""

from scipy import optimize
import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
import ntpath
import math
import datetime
import pkg_resources


def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size."""
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if len(x) < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[2*x[0]-x[window_len-1::-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


def term_sum(t, ups, ua, n, t0, m, sep):
    """Calculates the terms of the summatory of the slab model"""

    v = 299.792458 / n
    if n > 1:
        A = 504.332889 - 2641.00214 * n + 5923.699064 * n**2 - 7376.355814 * n**3 + \
            5507.53041 * n**4 - 2463.357945 * n**5 + \
            610.956547 * n**6 - 64.8047 * n**7
    else:
        A = 3.084635 - 6.531194 * n + 8.357854 * \
            n**2 - 5.082751 * n**3 + 1.171382 * n**4

    D = 1 / (3 * (ups + ua))
    ze = 2 * A * D
    z1 = sep * (1 - 2 * m) - 4 * m * ze - (1 / ups)
    z2 = sep * (1 - 2 * m) - (4 * m - 2) * ze + (1 / ups)
    val1 = 4 * D * v * (t + t0)
    return z1 * np.exp(-z1**2 / val1) - z2 * np.exp(-z2**2 / val1)


def funcion_teo_ex_slab(t, ups, ua, n, t0, sep, ro, dipolos=20):
    """Calculates the theoretical slab model (Contini)"""
    v = 299.792458 / n
    if n > 1:
        A = 504.332889 - 2641.00214 * n + 5923.699064 * n**2 - 7376.355814 * n**3 + \
            5507.53041 * n**4 - 2463.357945 * n**5 + \
            610.956547 * n**6 - 64.8047 * n**7
    else:
        A = 3.084635 - 6.531194 * n + 8.357854 * \
            n**2 - 5.082751 * n**3 + 1.171382 * n**4
    suma = sum(term_sum(t, ups, ua, n, t0, i, sep)
               for i in range(-dipolos // 2, dipolos // 2 + 1))
    D = 1 / (3 * (ups + ua))
    val2 = 4 * math.pi * D * v
    pow1 = val2 ** 1.5
    pow2 = (t + t0) ** 2.5
    pow3 = ro ** 2
    val1 = 4 * D * v * (t + t0)
    val3 = -(ua * v * (t + t0)) - pow3 / val1
    pow4 = np.exp(val3)
    return (pow4 / (2 * pow1 * pow2)) * suma


def funcion_teo_refl(t, ups, ua, n, t0, ro):
    """Calculates the theoretical semi-infinite reflectance model (Contini)"""
    v = 299.792458 / n
    if n > 1:
        A = 504.332889 - 2641.00214 * n + 5923.699064 * n**2 - 7376.355814 * n**3 + \
            5507.53041 * n**4 - 2463.357945 * n**5 + \
            610.956547 * n**6 - 64.8047 * n**7
    else:
        A = 3.084635 - 6.531194 * n + 8.357854 * \
            n**2 - 5.082751 * n**3 + 1.171382 * n**4
    z0 = 1 / ups
    D = 1 / (3 * (ups + ua))
    ze = 2 * A * D
    val1 = ua * v * (t + t0)
    val2 = (ro ** 2) / (4 * D * v * (t + t0))
    val3 = 4 * math.pi * D * v
    val4 = 2 * ze + z0
    pow1 = math.exp(-val1 - val2)
    pow2 = val3 ** 1.5
    pow3 = (t + t0) ** 2.5
    pow4 = z0 ** 2
    pow5 = math.exp(-pow4 / (4 * D * v * (t + t0)))
    pow6 = val4 ** 2
    pow7 = math.exp(-pow6 / (4 * D * v * (t + t0)))
    return ((-pow1) / (2 * pow2 * pow3)) * ((-z0 * pow5) - (val4 * pow7))


def funcion_fiteo_refl(t, ups, ua, back, t0, n, ro, instru, temp_data, first_nonzero_data, last_nonzero_data):
    """Calculates the convoluted function to fit - Semi-infinite model"""
    array_teo = funcion_teo_refl(temp_data, ups, ua, n, t0, ro)
    array_teo = array_teo / np.trapz(array_teo, temp_data)
    array_conv = np.convolve(instru, array_teo, 'full')
    array_conv.resize(len(temp_data))
    array_conv = array_conv / np.trapz(array_conv, temp_data)
    array_conv[:first_nonzero_data] = 0
    array_conv[last_nonzero_data:] = 0
    return np.interp(t, temp_data, array_conv) + back


def funcion_fiteo_slab(t, ups, ua, back, t0, n, ro, sep, instru, temp_data, first_nonzero_data, last_nonzero_data):
    """Calculates the convoluted function to fit - Slab model"""
    array_teo = funcion_teo_ex_slab(temp_data, ups, ua, n, t0, sep, ro)
    array_teo = array_teo / np.trapz(array_teo, temp_data)
    array_conv = np.convolve(instru, array_teo, 'full')
    array_conv.resize(len(temp_data))
    array_conv = array_conv / np.trapz(array_conv, temp_data)
    array_conv[:first_nonzero_data] = 0
    array_conv[last_nonzero_data:] = 0
    return np.interp(t, temp_data, array_conv) + back


def process_data(data_read, instru_read, skip, gain, smoothStr):
    """ Process the data files

    Load data from text file and normalizes it. It also create the temporal data if necesary        

    input:
        data_read: Experimental data filename
        instru_read: Response data filename
        skip: Number of lines to skip in the header of the files
    """
    # print (data_read, instru_read)
    data = np.genfromtxt(data_read, comments='*',
                         skip_header=int(skip))  # Read data from file
    # print(len(data))
    instru = np.genfromtxt(instru_read, comments='*',
                           skip_header=int(skip))  # Read data from file
    if smoothStr > 0:
        data = smooth(data, window_len=smoothStr)
        instru = smooth(instru, window_len=smoothStr)
        
    instru = instru / instru.sum()  # Normalize response function to area=1
    # fm.data = fm.data / fm.smooth(fm.data).max() #Normalize experimental data with maximum smoothed version
    max_temp = 50./gain  # TCSPC temporal window size: 50ns/gain
    # Generate temporal x-axis array
    temp_data = np.arange(0, max_temp, max_temp/len(data))
    # Normalize experimental data with area
    data = data / np.trapz(data, temp_data)
    # print(temp_data, data)
    return temp_data, data, instru


def write_results(idx, data_read, fileExt, type_fit, results, ups_init, ua_init, t0_init, sep, ro, file_instru):
    """ Write the results files

    Write the results of the fitting in text files. It create two files: one with the fitted curve, 
    and one with the fitting parameters and the resulting values.

    input:
        idx: Index of the fitting to save. Key of the results dictionary.
        data_read: Experimental data filename. Used as the root of the exit filename.
        fileExt: Exit name modifier. Attached to the exit filename.
    """
    basename = ntpath.basename(data_read)
    # Experimental and fitted curves
    # Filename [DataFileName]_[fileExt]_curves.dat
    fitName = str(data_read) + "_" + fileExt + "_" + "curves.dat"
    fit_file = open(fitName, "w")
    fit_file.write("t\t Fit\t Exp.\n")
    for i in range((results[idx][6]).size):
        fit_file.write(str(results[idx][8][i]) + "\t" + str(results[idx]
                       [6][i]) + "\t" + str(results[idx][7][i]) + "\n")
    fit_file.close()
    # Result summary
    # Filename [DataFileName]_[fileExt]_results.dat
    resultsName = str(data_read) + "_" + fileExt + "_" + "results.dat"
    results_file = open(resultsName, "w")
    results_file.write(
        "Fitting date: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
    results_file.write("Entry data: \n")
    results_file.write("Fit type: ")
    if type_fit == 1:  # Slab
        results_file.write("Slab \n")
        results_file.write("Thickness: " + str(sep) + "\n")
    elif type_fit == 2:  # Semi-infinte
        results_file.write("Semi infinite \n")
    results_file.write("Separation source-detector: " + str(ro) + "\n")
    results_file.write("Data file: " + str(basename) + "\n")
    results_file.write("Response file: " +
                       str(ntpath.basename(file_instru)) + "\n")
    results_file.write("Initial values: ups = " + str(ups_init) +
                       " ua = " + str(ua_init) + " t0 = " + str(t0_init) + "\n\n")
    results_file.write("Results: \n")
    results_file.write("Fitted values: ups = " + str(results[idx][1]) + " ua = " + str(
        results[idx][2]) + " t0 = " + str(results[idx][3]) + " baseline corr= " + str(results[idx][4]) + "\n")
    results_file.write("Norm: " + str(results[idx][5]) + "\n")


def FitFunction(file_instru, file_data, file_result, ups_init, ua_init, t0_init, back_init, type_fit, t0_fixed, sep, ro, n_ref, file_head, gain, cutThr=0, smoothStr = 0, writeExternalResults = True):
    """ Main fitting function

    After pressing "Fit" button, this functions calls the curve_fit function with the apropiate model
    and appends the fitting results to the results dictionary
    """
    # print (file_instru, file_data, file_result, ups_init, ua_init, t0_init)

    results = dict()

    index = 0  # Dictionary key, number of fit

    print("Input values: ups = {}, ua = {}, t0 = {}, back = {}, sep = {}, ro = {}, n = {}\n".format(
        ups_init, ua_init, t0_init, back_init, sep, ro, n_ref))

    if t0_fixed == 0:
        init_vals = ups_init, ua_init, t0_init, back_init
    elif t0_fixed == 1:
        init_vals = ups_init, ua_init, back_init

    for filename in file_data:  # Loop through the experimental files.
        index = index+1
        temp_data, data, instru = process_data(
            filename, file_instru, file_head, gain, smoothStr)

        if cutThr == 0:
            first_nonzero_data = np.nonzero(data)[0][0]
            # Calculate position of first nonzero value in data
            last_nonzero_data = np.nonzero(data)[0][-1]
        else:
            cutValue = np.max(data) * cutThr
            dataCutted = data
            dataCutted[data < cutValue] = 0
            first_nonzero_data = np.nonzero(dataCutted)[0][0]
            last_nonzero_data = np.nonzero(dataCutted)[0][-1]

        print("Fitting {} ...".format(ntpath.basename(filename)))
        print("Cut values: {}...{}".format(
            first_nonzero_data, last_nonzero_data))

        # array_teo = fm.funcion_teo_ex_slab(temp_data, ups_init, ua_init, n_ref, t0_init, sep, ro)
        # return temp_data, array_teo

        if type_fit == 1:  # Slab
            if t0_fixed == 0:
                popt, pcov = optimize.curve_fit(lambda t, ups, ua, back, t0: funcion_fiteo_slab(
                    t, ups, ua, back, t0, n_ref, ro, sep, instru, temp_data, first_nonzero_data, last_nonzero_data), temp_data, data, init_vals,
                    bounds=([0, 0, 0, 0], [2, 0.1, 1, 1]))
                Y = funcion_fiteo_slab(
                    temp_data, popt[0], popt[1], popt[2], popt[3], n_ref, ro, sep, instru, temp_data, first_nonzero_data, last_nonzero_data)
                print("Fitted values: ")
                print("ups: {}, ua: {}, t0: {}, baseline correction: {}".format(
                    popt[0], popt[1], popt[2], popt[3]))
                norm = ((Y-data)**2).sum()
                print("Norm: ", norm)
                print("_____________________")
            elif t0_fixed == 1:
                popt, pcov = optimize.curve_fit(lambda t, ups, ua, back: funcion_fiteo_slab(
                    t, ups, ua, back, t0_init, n_ref, ro, sep, instru, temp_data, first_nonzero_data, last_nonzero_data), temp_data, data, init_vals,
                    bounds=([0, 0, 0], [2, 0.1, 1]))
                Y = funcion_fiteo_slab(
                    temp_data, popt[0], popt[1], popt[2], t0_init, n_ref, ro, sep, instru, temp_data, first_nonzero_data, last_nonzero_data)
                print("Fitted values: ")
                print("ups: {}, ua: {}, baseline correction: {}".format(
                    popt[0], popt[1], popt[2]))
                norm = ((Y-data)**2).sum()
                print("Norm: ", norm)
                print("_____________________")
        elif type_fit == 2:  # Semi-infinite
            if t0_fixed == 0:
                popt, pcov = optimize.curve_fit(lambda t, ups, ua, back, t0: funcion_fiteo_refl(
                    t, ups, ua, back, t0, n_ref, ro, sep, instru, temp_data, first_nonzero_data, last_nonzero_data), temp_data, data, init_vals,
                    bounds=([0, 0, 0, 0], [2, 0.1, 1, 1]))
                Y = funcion_fiteo_refl(
                    temp_data, popt[0], popt[1], popt[2], popt[3], n_ref, ro, sep, instru, temp_data, first_nonzero_data, last_nonzero_data)
                print("Fitted values: ")
                print("ups: {}, ua: {}, t0: {}, baseline correction: {}".format(
                    popt[0], popt[1], popt[2], popt[3]))
                norm = ((Y-data)**2).sum()
                print("Norm: ", norm)
                print("_____________________")
            elif t0_fixed == 1:
                popt, pcov = optimize.curve_fit(lambda t, ups, ua, back: funcion_fiteo_refl(
                    t, ups, ua, back, t0_init, n_ref, ro, sep, instru, temp_data, first_nonzero_data, last_nonzero_data), temp_data, data, init_vals,
                    bounds=([0, 0, 0], [2, 0.1, 1]))
                Y = funcion_fiteo_refl(
                    temp_data, popt[0], popt[1], popt[2], t0_init, n_ref, ro, sep, instru, temp_data, first_nonzero_data, last_nonzero_data)
                print("Fitted values: ")
                print("ups: {}, ua: {}, baseline correction: {}".format(
                    popt[0], popt[1], popt[2]))
                norm = ((Y-data)**2).sum()
                print("Norm: ", norm)
                print("_____________________")

        # Append results to the results dictionary
        if t0_fixed == 0:
            results.setdefault(index, []).append(
                ntpath.basename(filename))  # 0
            results.setdefault(index, []).append(popt[0])  # 1, ups
            results.setdefault(index, []).append(popt[1])  # 2, ua
            results.setdefault(index, []).append(popt[2])  # 3, back
            results.setdefault(index, []).append(popt[3])  # 4, t0
            results.setdefault(index, []).append(norm)  # 5
            results.setdefault(index, []).append(Y)  # 6
            results.setdefault(index, []).append(temp_data)  # 7
            results.setdefault(index, []).append(data)  # 8
            results.setdefault(index, []).append(instru)  # 9

        if t0_fixed == 1:
            results.setdefault(index, []).append(
                ntpath.basename(filename))  # 0
            results.setdefault(index, []).append(popt[0])  # 1
            results.setdefault(index, []).append(popt[1])  # 2
            results.setdefault(index, []).append(t0_init)  # 3
            results.setdefault(index, []).append(popt[2])  # 4
            results.setdefault(index, []).append(norm)  # 5
            results.setdefault(index, []).append(Y)  # 6
            results.setdefault(index, []).append(temp_data)  # 7
            results.setdefault(index, []).append(data)  # 8
            results.setdefault(index, []).append(instru)  # 9

        if writeExternalResults:
            write_results(index, filename, file_result, type_fit, results,
                          ups_init, ua_init, t0_init, sep, ro, file_instru)
        
    return results

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:21:33 2014

@author: Nicolas Abel Carbone / ncarbone@exa.unicen.edu.ar

CLI implementation (TODO)
"""

#!/usr/bin/env python

import pygtk
pygtk.require('2.0')
import FiteoMath as fm
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

init_ups = 1
init_ua = 0.01
init_t0 = 0.2
init_back = 1 
file_data = []

def user_input_InitVals():  
    init_ups = raw_input("ups inicial: ")
    init_ua = raw_input("ua inicial: ")
    init_t0 = raw_input("t0 inicial: ")
    init_back = raw_input("Background inicial: ")
    return float(init_ups), float(init_ua), float(init_t0), float(init_back)
    
def user_input_ExpData():
    global file_data 
    file_data = raw_input("Archivo de datos: ")
    file_instru = raw_input("Archivo respuesta: ")    
    fm.data = np.loadtxt(file_data)
    fm.instru = np.loadtxt(file_instru)
    fm.instru = fm.instru / fm.instru.sum()
    fm.data = fm.data / fm.smooth(fm.data).max()
    
    fm.sep = float(raw_input("sep: "))
    fm.ro = float(raw_input("ro: "))
    ganancia = float(raw_input("ganancia: "))
    max_temp = 50/ganancia
    fm.temp_data = np.arange(0,max_temp,max_temp/float(fm.data.size))
    
def write_results(res, fileExt):
    fileName = str(file_data) + fileExt
    fit_file = open(fileName, "w")
    for item in res:
        fit_file.write("%s\n" % item)
    
def main():
    user_input_ExpData()    
    init_vals = user_input_InitVals()# init_ups, init_ua, init_t0, init_back # 1, 0.01, 0.2, 1
    print "Valores iniciales: ", init_vals
    popt, pcov = optimize.curve_fit(fm.funcion_fiteo_slab, fm.temp_data, fm.data, init_vals)
    print "Valores ajuste: ups: ", popt[0], "ua: ", popt[1], "t0: ", popt[2], "back: ", popt[3]
    Y = fm.funcion_fiteo_slab(fm.temp_data,popt[0],popt[1],popt[2],popt[3])
    plt.plot(Y)
    plt.plot(fm.data)
    write_results(Y, "_curva.dat")
    
if __name__ == "__main__":
    main()
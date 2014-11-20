# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:11:00 2014

@author: Nicolas Abel Carbone / ncarbone@exa.unicen.edu.ar

GUI Glade implementation
"""

#!/usr/bin/env python

#imports
import pygtk
pygtk.require('2.0')
import gtk
import gtk.glade
#from gi.repository import gtk
#from gi.repository import gtk.glade

import FiteoMath as fm
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import ntpath
import math
import datetime

#global variable declarations with default values
file_instru = []  #Response data filename declaration
file_data = []  #Experimental data filename declaration
file_head = 9  #Number of header lines in data and response files (defualt for TCSPC=9)
file_result = "" #Default addon to result filenames (empty)

ups_init = 1  #Default initial ups
ua_init = 0.01  #Default initial ua
t0_init = 0.2 #Default initial t0
back_init = 1 #Default initial background

sep = 30  #Slab thickness
ro = 0  #Separation between the source and the optical axis 
gain = 2  #TCSPC temporal gain
n_ref = 1.4  #Refraction index

type_fit = 1  #Fitting type. 1: slab; 2: semi-infinite
t0_fixed = 0 #Fix t0?. 0: No; 1: Yes

results = dict()  #Dictionary with the fitting results
index = 0 #Dictionary key, number of fit
 
def process_data(data_read, instru_read, skip):
    """ Process the data files
        
    Load data from text file and normalizes it. It also create the temporal data if necesary        
        
    input:
        data_read: Experimental data filename
        instru_read: Response data filename
        skip: Number of lines to skip in the header of the files
    """
    global data
    print instru_read
    fm.data = np.genfromtxt(data_read, comments = '*', skip_header = int(skip)) #Read data from file
    fm.instru = np.genfromtxt(instru_read, comments = '*', skip_header = int(skip)) #Read data from file
    fm.instru = fm.instru / fm.instru.sum() #Normalize response function to area=1
    fm.data = fm.data / fm.smooth(fm.data).max() #Normalize experimental data with maximum smoothed version
    fm.max_temp = 50./gain #TCSPC temporal window size: 50ns/gain
    fm.temp_data = np.arange(0,fm.max_temp,fm.max_temp/float(fm.data.size)) #Generate temporal x-axis array    
   
def write_results(idx, data_read, fileExt):
    """ Write the results files

    Write the results of the fitting in text files. It create two files: one with the fitted curve, 
    and one with the fitting parameters and the resulting values.
        
    input:
        idx: Index of the fitting to save. Key of the results dictionary.
        data_read: Experimental data filename. Used as the root of the exit filename.
        fileExt: Exit name modifier. Attached to the exit filename.
    """ 
    basename = ntpath.basename(data_read)
    #Experimental and fitted curves
    fitName = str(data_read) + "_" + fileExt + "_" + "curves.dat" #Filename [DataFileName]_[fileExt]_curves.dat
    fit_file = open(fitName, "w")
    fit_file.write("t\t Fit\t Exp.\n")
    for i in range((results[idx][6]).size):
        fit_file.write(str(results[idx][8][i]) + "\t" + str(results[idx][6][i]) + "\t" + str(results[idx][7][i]) + "\n")
    fit_file.close()
    #Result summary
    resultsName = str(data_read) + "_" + fileExt + "_" + "results.dat" #Filename [DataFileName]_[fileExt]_results.dat
    results_file = open(resultsName, "w")
    results_file.write("Fitting date: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
    results_file.write("Entry data: \n")        
    results_file.write("Fit type: ")
    if type_fit == 1: #Slab
        results_file.write("Slab \n")
        results_file.write("Thickness: " + str(sep) + "\n")
    elif type_fit == 2: #Semi-infinte
        results_file.write("Semi infinite \n")
    results_file.write("Separation source-detector: " + str(ro) + "\n")
    results_file.write("Data file: " + str(basename) + "\n")
    results_file.write("Response file: " + str(ntpath.basename(file_instru)) + "\n")
    results_file.write("Initial values: ups = " + str(ups_init) + " ua = " + str(ua_init) + " t0 = " + str(t0_init) + "\n\n")
    results_file.write("Results: \n")
    results_file.write("Fitted values: ups = " + str(results[idx][1]) + " ua = " + str(results[idx][2]) + " t0 = " + str(results[idx][3]) + " baseline corr= " + str(results[idx][4]) + "\n")
    results_file.write("Norm: " + str(results[idx][5]) + "\n")     
    
class MainWindow:
    def __init__(self):
        
        #Declaration and build of interface elements
        builder = gtk.Builder()
        builder.add_from_file("GTKGlade.glade")
        builder.connect_signals(self)        
        
        self.MainWindow = builder.get_object("MainWindow")
        self.MainWindow.show()
        
        self.SelectSlab = builder.get_object("SelectSlab")
        self.SelectSlab.show()
        
        self.SelectSemiInf = builder.get_object("SelectSemiInf")
        self.SelectSemiInf.show()
        
        self.FitButton = builder.get_object("FitButton")
        self.FitButton.show()
        
        self.PlotButton = builder.get_object("PlotButton")
        self.PlotButton.show()
        
        self.ClearButton = builder.get_object("ClearButton")
        self.ClearButton.show()
        
        self.ExportButton = builder.get_object("ExportButton")
        self.ExportButton.show()             

        self.FileData = builder.get_object("FileData")
        self.FileData.show()
        
        self.FileDataButton = builder.get_object("FileDataButton")
        self.FileDataButton.show()
        
        self.FileInstru = builder.get_object("FileInstru")
        self.FileInstru.show()
        
        self.FileInstruButton = builder.get_object("FileInstruButton")
        self.FileInstruButton.show()
        
        self.FileHead = builder.get_object("FileHead")
        self.FileHead.show()   
        self.FileHead.set_text(str(file_head))
        
        self.FileResult = builder.get_object("FileResult")
        self.FileResult.show()
        self.FileResult.set_text(file_result)
        
        self.Sep = builder.get_object("Sep")
        self.Sep.show()
        self.Sep.set_text(str(sep))        
        
        self.Ro = builder.get_object("Ro")
        self.Ro.show()
        self.Ro.set_text(str(ro))
        
        self.Gain = builder.get_object("Gain")
        self.Gain.show()
        self.Gain.set_text(str(gain))
        
        self.NIndex = builder.get_object("NIndex")
        self.NIndex.show()
        self.NIndex.set_text(str(n_ref))
        
        self.upsInit = builder.get_object("upsInit")
        self.upsInit.show()
        self.upsInit.set_text(str(ups_init))
        
        self.uaInit = builder.get_object("uaInit")
        self.uaInit.show()
        self.uaInit.set_text(str(ua_init))
        
        self.t0Init = builder.get_object("t0Init")
        self.t0Init.show()
        self.t0Init.set_text(str(t0_init))
        
        self.t0FitCheck = builder.get_object("t0FitCheck")
        self.t0FitCheck.show()
        
        #Results table
        self.Results = builder.get_object("Results")
        self.Results.show()
        self.ColumnData = gtk.TreeViewColumn("Id", gtk.CellRendererText(), text=0)
        self.Results.append_column(self.ColumnData)        
        self.ColumnData = gtk.TreeViewColumn("Data file", gtk.CellRendererText(), text=1)
        self.Results.append_column(self.ColumnData)
        self.ColumnSuffix = gtk.TreeViewColumn("Suffix", gtk.CellRendererText(), text=2)
        self.Results.append_column(self.ColumnSuffix)
        self.ColumnType = gtk.TreeViewColumn("Type", gtk.CellRendererText(), text=3)
        self.Results.append_column(self.ColumnType)
        self.ColumnUps = gtk.TreeViewColumn("Ups", gtk.CellRendererText(), text=4)
        self.Results.append_column(self.ColumnUps)
        self.ColumnUa = gtk.TreeViewColumn("Ua", gtk.CellRendererText(), text=5)
        self.Results.append_column(self.ColumnUa)
        self.ColumnT0 = gtk.TreeViewColumn("T0", gtk.CellRendererText(), text=6)
        self.Results.append_column(self.ColumnT0)
        self.ColumnUpsVar = gtk.TreeViewColumn("Ups StdD", gtk.CellRendererText(), text=7)
        self.Results.append_column(self.ColumnUpsVar)
        self.ColumnUaVar = gtk.TreeViewColumn("Ua StdD", gtk.CellRendererText(), text=8)
        self.Results.append_column(self.ColumnUaVar)
        self.ColumnT0Var = gtk.TreeViewColumn("T0 StdD", gtk.CellRendererText(), text=9)
        self.Results.append_column(self.ColumnT0Var)
        self.ColumnNorm = gtk.TreeViewColumn("Norm", gtk.CellRendererText(), text=10)
        self.Results.append_column(self.ColumnNorm)
        self.ResultStore = gtk.ListStore(int, str, str, str, float, float, float, float, float, float, float)
        self.Results.set_model(self.ResultStore)
        
        #File open/save dialogs
        self.FileChooserDialogMulti = gtk.FileChooserDialog(title=None,action=gtk.FILE_CHOOSER_ACTION_OPEN,
                                             buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,gtk.STOCK_OPEN,gtk.RESPONSE_OK))
        self.FileChooserDialogMulti.set_select_multiple(True)
        
        self.FileChooserDialog = gtk.FileChooserDialog(title=None,action=gtk.FILE_CHOOSER_ACTION_OPEN,
                                             buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,gtk.STOCK_OPEN,gtk.RESPONSE_OK))
        self.FileChooserDialog.set_select_multiple(False)
        
        self.FileChooserDialogSave = gtk.FileChooserDialog(title=None,action=gtk.FILE_CHOOSER_ACTION_SAVE,
                                             buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,gtk.STOCK_SAVE,gtk.RESPONSE_OK))
        self.FileChooserDialogSave.set_select_multiple(False)
   
    #Declaration and implementation of interface functions     
    def on_MainWindow_destroy(self, widget, data = None):
        gtk.main_quit()
        self.FileChooserDialog.destroy()
        self.FileChooserDialogMulti.destroy()
        
    def on_FileChooserDialog_destroy(self, widget, data = None):
        self.FileChooserDialog.destroy()

    def SelectSlab_toggled_cb(self, widget, data = None):
        global type_fit
        print "Slab" 
        type_fit = 1
        self.Sep.set_editable(True)
    
    def SelectSemiInf_toggled_cb(self, widget, data = None):
        global type_fit
        print "SemiInf"        
        type_fit = 2
        self.Sep.set_text("N/A")
        self.Sep.set_editable(False)
 
    def FileResult_changed_cb(self, widget, data=None):
        global file_result
        file_result = self.FileResult.get_text()
        
    def FileHead_changed_cb(self, widget, data=None):
        global file_head
        file_head = float(self.FileHead.get_text())
        
    def Sep_changed_cb(self, widget, data=None):
        global sep
        sep = float(self.Sep.get_text())
    
    def Ro_changed_cb(self, widget, data=None):
        global ro
        ro = float(self.Ro.get_text())
    
    def Gain_changed_cb(self, widget, data=None):
        global gain
        gain = float(self.Gain.get_text())
    
    def NIndex_changed_cb(self, widget, data=None):
        global n_ref
        n_ref = float(self.NIndex.get_text())
    
    def upsInit_changed_cb(self, widget, data=None):
        global ups_init
        ups_init = float(self.upsInit.get_text())
        
    def uaInit_changed_cb(self, widget, data=None):
        global ua_init
        ua_init = float(self.uaInit.get_text())
        
    def t0Init_changed_cb(self, widget, data=None):
        global t0_init
        t0_init = float(self.t0Init.get_text())
        
    def t0FitCheck_toggled_cb(self, widget, data=None):
        global t0_fixed
        if self.t0FitCheck.get_active():
            t0_fixed = 1
        else:
            t0_fixed = 0
        print t0_fixed
        
    
    def FileDataButton_clicked_cb(self, widget, data=None): 
        """ File Data input
        
        It opens a Load file dialog and recovers the experimental data filename
        """        
        global file_data
        global OpenDialogData
        response = self.FileChooserDialogMulti.run()
        if response == gtk.RESPONSE_OK:
            OpenDialogData = True
            file_data[:] = [] #Empty file list
            #print self.FileChooserDialog.get_filenames(), 'selected'
            text = str(len(self.FileChooserDialogMulti.get_filenames())) + " " + "files loaded."           
            self.FileData.set_text(text)
            file_data=self.FileChooserDialogMulti.get_filenames()
        elif response == gtk.RESPONSE_CANCEL:
            print 'Closed, no files selected'
        self.FileChooserDialogMulti.hide()
        
    def FileInstruButton_clicked_cb(self, widget, data=None):
        """ File Instru input
        
        It opens a Load file dialog and recovers the response data filename
        """ 
        global file_instru
        response = self.FileChooserDialog.run()
        if response == gtk.RESPONSE_OK:
            text = str(ntpath.basename(self.FileChooserDialog.get_filename()))
            self.FileInstru.set_text(text)
            file_instru = self.FileChooserDialog.get_filename()
        elif response == gtk.RESPONSE_CANCEL:
            print 'Closed, no files selected'
        self.FileChooserDialog.hide()

    def PlotButton_clicked_cb(self, widget, data=None):
        """ Plot function
        
        Plot the selected results
        """ 
        selection = self.Results.get_selection()
        model, treeiter = selection.get_selected()
        key = model[treeiter][0]
        first_nonzero_data = np.nonzero(results[key][7])[0][0] #Calculate position of first nonzero value in data
        last_nonzero_data = np.nonzero(results[key][7])[0][-1] #Calculate position of last nonzero value in data         
        fig = plt.figure() #Main figure
        gs = plt.GridSpec(2, 1,height_ratios=[2,1]) #2x1 grid       
        fit = fig.add_subplot(gs[0]) #Fitting curves sub-figure
        res = fig.add_subplot(gs[1]) #Residuals sub-figure 
        line, = fit.plot(results[key][8][first_nonzero_data:last_nonzero_data], 
                         results[key][6][first_nonzero_data:last_nonzero_data], color='blue', lw=2) #Fitted curve plot
        line, = fit.plot(results[key][8][first_nonzero_data:last_nonzero_data], 
                         results[key][7][first_nonzero_data:last_nonzero_data], color='red', lw=2) #Experimental data plot
        line, = res.plot(results[key][8][first_nonzero_data:last_nonzero_data], 
                         (results[key][6][first_nonzero_data:last_nonzero_data]-
                         results[key][7][first_nonzero_data:last_nonzero_data])/
                         results[key][6][first_nonzero_data:last_nonzero_data], color='black', lw=1) #Residual plot
        fit.set_yscale('log') #Set Fitting figure y-axis to logarithmic
        res.set_ylim([-0.5,0.5]) #Set residual y-axis scale
        plt.show() #Show plots

    def ExportButton_clicked_cb(self, widget, data=None):
        """ Export function
        
        Export to an Ascii file the selected results
        """
        global results
        response = self.FileChooserDialogSave.run()
        if response == gtk.RESPONSE_OK:
            fileexportName = self.FileChooserDialogSave.get_filename()
            export_file = open(fileexportName, "w")
            export_file.write("Id" + "\t" + "Exp. File" + "\t" + "ups" + "\t" + "ua" + "\t" + "t0" + "\t" + "norm" + "\n")
            for k, v in results.items():
                export_file.write(str(k) + "\t" + str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\t" + str(v[5]) + "\n")
            export_file.close()
        elif response == gtk.RESPONSE_CANCEL:
            print 'Closed, no files selected'
        self.FileChooserDialogSave.hide()
    
    def ClearButton_clicked_cb(self, widget, data=None):
        """ Clear list of results
        """    
        results.clear()
        self.ResultStore.clear()
        
    def FitButton_clicked_cb(self, widget, data=None):
        """ Main fitting function
        
        After pressing "Fit" button, this functions calls the curve_fit function with the apropiate model
        and appends the fitting results to the results dictionary
        """        
        global results
        global file_instru
        global index        
        print file_instru, file_data, file_result, ups_init, ua_init, t0_init
        if not file_instru: #If file_instru is empty, the file chooser dialog was not opened (or was canceled). Use the textbox.
            file_instru = self.FileInstru.get_text()
        if not file_data: #If file_data is empty, the file chooser dialog was not opened (or was canceled). Use the textbox.
            file_data.append(self.FileData.get_text())
        fm.sep = sep
        fm.ro = ro
        fm.n_ref = n_ref
        if t0_fixed == 0:
            init_vals = ups_init, ua_init, t0_init, back_init        
        elif t0_fixed == 1:
            init_vals = ups_init, ua_init, back_init  
        
        for filename in file_data: #Loop through the experimental files.             
            index=index+1            
            process_data(filename, file_instru, file_head)
            fm.pre_calcs()
            if type_fit == 1: #Slab
                if t0_fixed == 0:
                    print str(type_fit) + "Slab, t0 not fixed"            
                    popt, pcov = optimize.curve_fit(fm.funcion_fiteo_slab, fm.temp_data, fm.data, init_vals)
                    Y = fm.funcion_fiteo_slab(fm.temp_data,popt[0],popt[1],popt[2],popt[3])
                elif t0_fixed == 1:
                    print str(type_fit) + "Slab, t0 fixed"            
                    popt, pcov = optimize.curve_fit(lambda t, ups, ua, back: fm.funcion_fiteo_slab(t, ups, ua, t0_init, back), fm.temp_data, fm.data, init_vals)
                    Y = fm.funcion_fiteo_slab(fm.temp_data,popt[0],popt[1],t0_init,popt[2])
            elif type_fit == 2: #Semi-infinite
                if t0_fixed == 0:
                    print str(type_fit) + "Semi-infinte, t0 not fixed"
                    popt, pcov = optimize.curve_fit(fm.funcion_fiteo_refl, fm.temp_data, fm.data, init_vals)
                    Y = fm.funcion_fiteo_refl(fm.temp_data,popt[0],popt[1],popt[2],popt[3])
                elif t0_fixed == 1:
                    print str(type_fit) + "Semi-infinte, t0 fixed"
                    popt, pcov = optimize.curve_fit(lambda t, ups, ua, back: fm.funcion_fiteo_refl(t, ups, ua, t0_init, back), fm.temp_data, fm.data, init_vals)
                    Y = fm.funcion_fiteo_refl(fm.temp_data,popt[0],popt[1],t0_init,popt[2])
            print "Fitted values: ", popt
            norm = ((Y-fm.data)**2).sum()           
            print "Norm: ", norm
            
            #Append the results to the Result Store, so they showed in the GUI table.
            if type_fit == 1:
                if t0_fixed == 0:
                    if type(pcov) != float: #Check for Inf variances (it pcov is float and not a list, it is probably Inf). They make the append fail.
                        self.ResultStore.append([index, ntpath.basename(filename), file_result, "Slab", popt[0], popt[1], popt[2], math.sqrt(pcov[0,0]), math.sqrt(pcov[1,1]), math.sqrt(pcov[2,2]), norm])
                    else:
                        self.ResultStore.append([index, ntpath.basename(filename), file_result, "Slab", popt[0], popt[1], popt[2], 0., 0., 0., norm])

                elif t0_fixed == 1:
                    if type(pcov) != float: #Check for Inf variances (it pcov is float and not a list, it is probably Inf). They make the append fail.
                        self.ResultStore.append([index, ntpath.basename(filename), file_result, "Slab", popt[0], popt[1], t0_init, math.sqrt(pcov[0,0]), math.sqrt(pcov[1,1]), math.sqrt(pcov[2,2]), norm])
                    else:
                        self.ResultStore.append([index, ntpath.basename(filename), file_result, "Slab", popt[0], popt[1], t0_init, 0., 0., 0., norm])

            elif type_fit == 2:
                if t0_fixed == 0:
                    if type(pcov) != float: #Check for Inf variances (it pcov is float and not a list, it is probably Inf). They make the append fail.
                        self.ResultStore.append([index, ntpath.basename(filename), file_result, "Semi-Inf", popt[0], popt[1], popt[2], math.sqrt(pcov[0,0]), math.sqrt(pcov[1,1]), math.sqrt(pcov[2,2]), norm])
                    else:
                        self.ResultStore.append([index, ntpath.basename(filename), file_result, "Semi-Inf", popt[0], popt[1], popt[2], 0., 0., 0., norm])
                elif t0_fixed == 1:
                    if type(pcov) != float: #Check for Inf variances (it pcov is float and not a list, it is probably Inf). They make the append fail.
                        self.ResultStore.append([index, ntpath.basename(filename), file_result, "Semi_Inf", popt[0], popt[1], t0_init, math.sqrt(pcov[0,0]), math.sqrt(pcov[1,1]), math.sqrt(pcov[2,2]), norm])
                    else:
                        self.ResultStore.append([index, ntpath.basename(filename), file_result, "Semi_Inf", popt[0], popt[1], t0_init, 0., 0., 0., norm])
 
            #Append results to the results dictionary
            if t0_fixed == 0:            
                results.setdefault(index, []).append(ntpath.basename(filename))         
                results.setdefault(index, []).append(popt[0]) 
                results.setdefault(index, []).append(popt[1])
                results.setdefault(index, []).append(popt[2])            
                results.setdefault(index, []).append(popt[3])
                results.setdefault(index, []).append(norm)            
                results.setdefault(index, []).append(Y)
                results.setdefault(index, []).append(fm.data)
                results.setdefault(index, []).append(fm.temp_data)
            
            if t0_fixed == 1:            
                results.setdefault(index, []).append(ntpath.basename(filename))         
                results.setdefault(index, []).append(popt[0]) 
                results.setdefault(index, []).append(popt[1])
                results.setdefault(index, []).append(t0_init)            
                results.setdefault(index, []).append(popt[2])
                results.setdefault(index, []).append(norm)            
                results.setdefault(index, []).append(Y)
                results.setdefault(index, []).append(fm.data)
                results.setdefault(index, []).append(fm.temp_data)
            
            write_results(index, filename, file_result)

if __name__ == "__main__":
	hwg = MainWindow()
	gtk.main()
#! python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:34:04 2017

@author: kbeckers V1 and V2; Malcolm Ross V3
"""

#GEOPHIRES v3.0
#build date: May 2022
#github address: https://github.com/malcolm-dsider/GEOPHIRES-X

import os
import sys
import logging
import logging.config
import Model
import numpy as np
from OptionList import EndUseOptions

def run_geophires(filename,dailyheatingdemandfromdGeo,makeplot):
    
    #set up logging.
    logging.basicConfig(filename="GEOPHIRES3_Logging.log",format='%(asctime)s %(message)s',filemode='w')
    
    logger = logging.getLogger('root')
    logger.setLevel("INFO")
    
    logger.info("Init " + str(__name__))
    
    #set the starting directory to be the directory that this file is in
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    #initiate the entire model
    model = Model.Model(filename)
    
    #read the parameters that apply to the model
    model.read_parameters()
    
    #assign daily heating demand to correct parameter
    model.surfaceplant.dailyheatingdemand.value = dailyheatingdemandfromdGeo
    
    #Calculate the entire model
    model.Calculate()
    
    #write the outputs, if requested
    model.outputs.PrintOutputs(model)
            
    #if the user has asked for it, copy the output file to the screen
    if model.outputs.printoutput:
        outputfile = "HDR.out"
        if len(sys.argv) > 2: outputfile = sys.argv[2]
        with open(outputfile,'r', encoding='UTF-8') as f:
            content = f.readlines()    #store all output in one long list
    
            #Now write each line to the screen
            for line in content: sys.stdout.write(line)
            
    #make district heating plot
    if model.surfaceplant.enduseoption.value == EndUseOptions.DISTRICT_HEATING and makeplot == 1:        
        model.outputs.MakeDistrictHeatingPlot(model)    
        
    logger.info("Complete "+ str(__name__) + ": " + sys._getframe().f_code.co_name)
    
    logging.shutdown()
    
    return model.economics.LCOH.value, model.wellbores.ProducedTemperature.value
    

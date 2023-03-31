#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 23:08:01 2023

@author: kmccabe
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

import GEOPHIRESv3c


#%% INGEST DATA

# agents
agents = pd.read_csv('Data/idaho_com_res_2014.csv')

# load and distribution
county_demand_profiles_df = pd.read_pickle('Data/county_demand_profiles_df.pkl')
county_total_demand_df = pd.read_pickle('Data/county_total_demand_df.pkl')
county_total_demand_df.drop(columns='demand_total_heat_in_bin_kwh', inplace=True)
distribution_df = pd.read_csv('Data/distribution.csv')

# fuel prices
ng_prices_df = pd.read_pickle('Data/ng_prices_df.pkl')
elec_prices_df = pd.read_pickle('Data/elec_prices_df.pkl')

# resources
hydrothermal_df = pd.read_csv('Data/resources_hydrothermal.csv')
egs_df = pd.read_csv('Data/resources_egs.csv')
all_resource_df = pd.concat([hydrothermal_df, egs_df], ignore_index=True)


#%% CONFIG INPUTS

# set resource parameters - resource_type must be 'hydrothermal' or 'egs'
resource_type = 'hydrothermal'
egs_target_temp_deg_c = 100.

# subset resource dataframe
resource_df = all_resource_df.loc[all_resource_df['resource_type'] == resource_type]

# for egs - add logic to filter resources to temperature closest to target temperature
if resource_type == 'egs':
    resource_df['diff'] = abs(resource_df['res_temp_deg_c'] - egs_target_temp_deg_c)
    resource_df = resource_df.sort_values('diff', ascending=True).groupby('source_resource_id').apply(pd.DataFrame.head, n=1).reset_index(drop=True)
    resource_df.drop(columns='diff', inplace=True)


#%% COMBINE INPUT DATA

# merge fuel costs
resource_df = resource_df.merge(ng_prices_df, how = 'left', on = ['tract_id_alias'])
resource_df = resource_df.merge(elec_prices_df, how = 'left', on = ['tract_id_alias'])

# merge county-level load totals
resource_df = resource_df.merge(distribution_df, how = 'left', on = ['county_id'])
resource_df = resource_df.merge(county_total_demand_df, how = 'left', on = ['county_id'])
resource_df = resource_df.merge(county_demand_profiles_df, how = 'left', on = ['county_id'])

# sanitize resource df - remove invalid entries
resource_df = resource_df.loc[resource_df['depth_m'] > 100]
resource_df = resource_df.loc[resource_df['res_temp_deg_c'] >= 50]
resource_df = resource_df.loc[resource_df['lifetime_resource_per_wellset_mwh'] > 0]


#%% GEOPHIRES SUPPLY CALCULATIONS

### -------------- ###
### Pre-processing ###
### -------------- ###
#specify constant parameters
SurfaceTemperature = 10     #deg.C (this is only used to calculate bottom hole temperature )
DoubletsPerGWh = 80         #This is a geothermal system sizing parameter: one doublet will be considered per specific annual heating demand
MakePlot = 0                #must be 0 or 1. 1 will create a heat supply plot in GEOPHIRES 
PrintToConsole = 0          #must be 0 or 1. 1 will print the GEOPHIRES results for each case to the console
UseAvgNGPrice = 1           #must be 0 or 1. 1 will take 30-yr average natural gas price, 0 will use year 1 price

# prepare LCOH arrays to store results
results = pd.DataFrame()

### -------------------------------------------------------- ###
### For-loop through all the tracts (or counties) considered ###
### -------------------------------------------------------- ###
for cty in resource_df['county_id'].unique().tolist():
    for i, row in resource_df.loc[resource_df['county_id']==cty].iterrows():
    
        # set resource type - 1 is hydrothermal, 2 is EGS
        ResourceType = 1 if row.loc['resource_type'] == 'hydrothermal' else 2
    
        #get the following parameter values for each census tract 
        ElectricityRate = row.loc['elec_price_dlrs_per_mwh']/1000.    #[$/kWh]
        NaturalGasRate = np.array(row.loc['ng_price_dlrs_per_mwh'])/1000.       #[$/kWh] ($8MMbtu = 0.0273 $/kWh)
        NaturalGasRate = np.mean(NaturalGasRate) if UseAvgNGPrice else NaturalGasRate[0]
        RoadLength = row.loc['distribution_total_m']/1000.                                               #[km] length of roads where the district heating system would be built. GEOPHIRES uses the Reber correlations internally to translate this to district heating system cost
        AnnualHeatingDemand = row.loc['demand_total_heat_mwh']/1000.                                     #[Gwh/year] this is the combined annual space and water heating demand for all the buildings in the community the district heating system will provide heat to
        
        #Next parameter is the normalized hourly heating demand profile over an entire year. Sum of elements equals one. Actual hourly heating demand is obtained by multiplying this array with AnnualHeatingDemand
        NormalizedHourlyHeatingProfile = np.array(row.loc['heat_demand_profile_mw'])/AnnualHeatingDemand/1000
    
        #get the following resource information for the local hydrothermal or EGS resource considered
        if ResourceType == 1: #hydrothermal
            #multiple hydrothermal resources can be present in the same tract.
            #In that case, we could either include a nested for-loop here or alternatively pick the "best" hydrothermal resource only.
            #To pick the best resource, I would first exlude all the ones less than 50C and with depth less than 100m. Of the ones left,
            #I would pick for now the one with the largest ResourceSize as it appears there are many with very small resource size.
            #the one with the largest resource size will often be the one with the largest temperature I think.
        
            ResourceTemperature = row.loc['res_temp_deg_c']                #[deg.C] this is the temperature of the hydrothermal resource [deg.C]
            Depth = row.loc['depth_m']/1000.                             #[km] for hydrothermal, this should be the depth of the hydrothermal resource
            ResourceSize = row.loc['n_wellsets_in_tract']*row.loc['lifetime_resource_per_wellset_mwh']                    #[MWh] (for hydrothermal only) maximum amount of heat we can extract over the lifetime
            #ResourceSize may be equal to n_wellsets_in_tract*lifetime_resource_per_wellset_mwh
        elif  ResourceType == 2: #EGS
            ResourceTemperature = row.loc['res_temp_deg_c']                #[deg.C] this is the temperature of the EGS resource [deg.C]
            Depth = row.loc['depth_m']/1000.                             #[km] I think for now we should pick the depth that gives us about 100deg.C
            ResourceSize = row.loc['n_wellsets_in_tract']*row.loc['lifetime_resource_per_wellset_mwh']
            
        
        ### ---------------------------------------------------------------------------------- ###
        ### Internal dGeo calculations in preparation of writing the GEOPHIRES input text file ###
        ### ---------------------------------------------------------------------------------- ###
        #calculate geothermal gradient
        GeothermalGradient = (ResourceTemperature-SurfaceTemperature)/Depth                 #[deg.C/km]
        
        #calculate daily heating demand from hourly heating demand
        DailyHeatDemand = np.zeros(365)
        year_hour = 0
        for day in range(0,365):                    # iterate through each day of the year
            D_sum = 0
            for hour in range(0,24):                # iterate through hours of each day
                D_sum += NormalizedHourlyHeatingProfile[year_hour]
                year_hour += 1
            DailyHeatDemand[day] = D_sum
        DailyHeatDemand = DailyHeatDemand*AnnualHeatingDemand*1000 #[MWh/day]
        
        #estimate optimal number of wells assuming DoubletsPerGWh per doublet and we pick at least 1 doublet. For now we only consider doublets. 
        NumberOfProductionWells = max(1,AnnualHeatingDemand//DoubletsPerGWh)
        NumberOfInjectionWells = NumberOfProductionWells
        
        #correct number of wells if resource size is too small (for hydrothermal).
        if ResourceType == 1:
            maximumnumberofwells = ResourceSize*1e6/20/8760/25/4200/(ResourceTemperature-40)
            NumberOfProductionWells = min(NumberOfProductionWells, round(maximumnumberofwells))
            NumberOfInjectionWells = NumberOfProductionWells
            
        #calculate if GDH will make sense before running GEOPHIRES. Heat demand needs to be high enough and hydrothermal resource needs to be warm and large enough  
        validforGEOPHIRES = 1 #assume a-priori simulation makes sense
        flag = 0
        if AnnualHeatingDemand < 10:
            validforGEOPHIRES = 0 #drilling even a doublet will not be competitive for too small communities (annual heating demand less than 10 GWh per year)
            flag = 1
        if ResourceTemperature < 50:
            validforGEOPHIRES = 0 #less than 50C, not much useful heat can be extracted for district heating
            flag = 2
        if ResourceSize < 2e5:
            validforGEOPHIRES = 0 #resource is too small to provide sufficient amount of heat
            flag = 3
        if NumberOfProductionWells == 0:
            validforGEOPHIRES = 0 #resource is too small to get at least 1 doublet
            flag = 4
        if Depth < 0.1:
            validforGEOPHIRES = 0 #resource is too small shallow
            flag = 5
    
        ### ------------------------------ ###
        ### Write GEOPHRES input text file ###
        ### ------------------------------ ###
        filename = str('GEOPHIRES_Input.txt')
        f = open(filename,'w')
        f.write('GEOPHIRES v3.0 Input File\n')
        f.write('***Subsurface Technical Parameters***\n')
        f.write('*************************************\n')
        f.write('Reservoir Model,4,						---Linear Thermal Drawdown Model\n')
        f.write('Reservoir Depth,%9.3f,						---[km]\n' % Depth)
        f.write('Drawdown Parameter,0.005,						---[-]\n')
        f.write('Number of Segments,1,						---[-]\n')
        f.write('Gradient 1,%9.3f,							---[deg.C/km]\n' %GeothermalGradient)
        f.write('Maximum Temperature,500,					---[deg.C]\n')
        f.write('Number of Production Wells,%9.0f,					---[-]\n' %NumberOfProductionWells)
        f.write('Number of Injection Wells,%9.0f,					---[-]\n' %NumberOfInjectionWells)
        f.write('Production Well Diameter,8,					---[inch]\n')
        f.write('Injection Well Diameter,8,					---[inch]\n')
        f.write('Ramey Production Wellbore Model,1,				---Should be 0 (disable) or 1(enable)\n')
        f.write('Production Wellbore Temperature Drop,0,\n')
        f.write('Injection Wellbore Temperature Gain,0,				---[deg.C]\n')
        f.write('Production Flow Rate per Well,50,				---[kg/s]\n')
        f.write('Reservoir Volume Option,4,					---Should be 1 2 3 or 4. See manual for option details\n')
        f.write('Reservoir Volume,1e9,				---m3\n')
        f.write('Reservoir Impedance,0.005,					---[GPa*s/m3]\n')
        f.write('Injection Temperature,40,					---[deg.C]\n')
        f.write('Maximum Drawdown,.3,						---Should be between 0 and 1 depending on redrilling\n')
        f.write('Reservoir Heat Capacity,825,					---[J/kg/K]\n')
        f.write('Reservoir Density,2730,						---[kg/m3]\n')
        f.write('Reservoir Thermal Conductivity,2.83,				---[W/m/K]\n')
        f.write('***Surface Technical Parameters***\n')
        f.write('**********************************\n')
        f.write('End-Use Option,8,						---District Heating System with Peaking boilers\n')
        f.write('Circulation Pump Efficiency,.8,					---[-]\n')
        f.write('Utilization Factor,.85,						---[-]\n')
        f.write('End-Use Efficiency Factor,.9,					---[-]\n')
        f.write('Surface Temperature,%9.3f,						---[deg.C]\n' %SurfaceTemperature)
        f.write('***District Heating System*\n')
        f.write('District Heating Demand Option,3,					---\n')
        f.write('District Heating Road Length,%9.3f,					--- km\n' %RoadLength)
        f.write('Peaking Boiler Efficiency,0.85,			`		--- Should be between 0 and 1, defaults to 0.85\n')
        f.write('Peaking Fuel Cost Rate,%9.4f, 					--- [$/kWh] Cost of natural gas for peak boiler use\n' %NaturalGasRate)
        f.write('***Financial Parameters***\n')
        f.write('**************************\n')
        f.write('Plant Lifetime,20,						---[years]\n')
        f.write('Economic Model,2,						---Standard Levelized Cost Model\n')
        f.write('Discount Rate,.07,						---[-]\n')
        f.write('Inflation Rate During Construction,0,				---[-]\n')
        f.write('***Capital and O&M Cost Parameters***\n')
        f.write('*************************************\n')
        f.write('Well Drilling and Completion Capital Cost Adjustment Factor,1,\n')
        if ResourceType == 1: #no stimulation required for hydrothermal resource
            f.write('Reservoir Stimulation Capital Cost Adjustment Factor,0, 	---[-] Use built-in correlations\n')
        elif ResourceType == 2: #EGS and stimulation is required
            f.write('Reservoir Stimulation Capital Cost Adjustment Factor,1, 	---[-] Use built-in correlations\n')
            
        f.write('Surface Plant Capital Cost Adjustment Factor,1,			---[-] Use built-in correlations\n')
        f.write('Field Gathering System Capital Cost Adjustment Factor,1,	---[-] Use built-in correlations\n')
        f.write('Exploration Capital Cost Adjustment Factor,0,			---[-] Use built-in correlations\n')
        f.write('Wellfield O&M Cost Adjustment Factor,1,				---[-] Use built in correlations\n')
        f.write('Surface Plant O&M Cost Adjustment Factor,1,			---[-] Use built-in correlations\n')
        f.write('Water Cost Adjustment Factor,1,					---[-] Use built-in correlations\n')
        f.write('Electricity Rate,%9.3f,						---[$/kWh]\n' %ElectricityRate)
        f.write('***Simulation Parameters***\n')
        f.write('***************************\n')
        f.write('Print Output to Console,%9.0f,					---should be either 0 (do not show) or 1 (show)\n' %PrintToConsole)
        f.write('Time steps per year,4,						---[-]\n')  
        f.close()
    
        ### ------------- ###
        ### Run GEOPHIRES ###
        ### ------------- ###
        if validforGEOPHIRES == 1:
            [LCOH, Toutput] = GEOPHIRESv3c.run_geophires(filename, DailyHeatDemand, MakePlot)
        else:
            LCOH = Toutput = -1
            
    
        ### ------------ ###
        ### Store Result ###
        ### ------------ ###
        out = {'resource_uid': row.loc['resource_uid'],
               'source_resource_id': row.loc['source_resource_id'],
               'resource_type': row.loc['resource_type'],
               'tract_id_alias': row.loc['tract_id_alias'],
               'county_id': row.loc['county_id'],
               'annual_heat_demand_gwh': AnnualHeatingDemand,
               'res_temp_deg_c': ResourceTemperature,
               'depth_km': Depth,
               'resource_size_mwh': ResourceSize,
               'lcoh_dlrs_per_mmbtu': LCOH,
               'flag': flag}
        temp_df = pd.DataFrame(data=out, index=[0])
        
        results = pd.concat([results, temp_df], ignore_index=True)
        
results.to_csv('Data/output.csv', index=False)

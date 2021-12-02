#!/bin/sh

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:01:00 2020

@author: ATeklesadik
"""
import time
import ftplib
import os
import sys
from datetime import datetime, timedelta
from sys import platform
import subprocess
import logging
import traceback
from pathlib import Path
from azure.storage.file import FileService
from azure.storage.file import ContentSettings

import pandas as pd
from pybufrkit.decoder import Decoder
import numpy as np
from geopandas.tools import sjoin
import geopandas as gpd
import click
import json
from climada.hazard import Centroids, TropCyclone,TCTracks
from climada.hazard.tc_tracks_forecast import TCForecast
from typhoonmodel.utility_fun import track_data_clean, Check_for_active_typhoon, Sendemail, \
    ucl_data, plot_intensity, initialize

if platform == "linux" or platform == "linux2": #check if running on linux or windows os
    from typhoonmodel.utility_fun import Rainfall_data
elif platform == "win32":
    from typhoonmodel.utility_fun import Rainfall_data_window as Rainfall_data
from typhoonmodel.utility_fun.forecast_process import Forecast
decoder = Decoder()

initialize.setup_logger()
logger = logging.getLogger(__name__)

@click.command()
@click.option('--path', default='./', help='main directory')
@click.option('--remote_directory', default=None, help='remote directory for ECMWF forecast data') #'20210421120000'
@click.option('--typhoonname', default=None, help='name for active typhoon')
@click.option('--typhoonname', default=None, help='name for active typhoon')
@click.option('--debug', is_flag=True, help='setting for DEBUG option')
def main(path,debug,remote_directory,typhoonname):
    initialize.setup_cartopy()
    start_time = datetime.now()
    ############## Defult variables which will be updated if a typhoon is active 
    landfall_time='NA'
    landfall_location_manucipality='NA'
    EAP_TRIGGERED='no'
    EAP_TRIGGERED_bool='false'
    print('---------------------AUTOMATION SCRIPT STARTED---------------------------------')
    print(str(start_time))
    #%% check for active typhoons
    print('---------------------check for active typhoons---------------------------------')
    print(str(start_time))
    remote_dir = remote_directory
    main_path=path
    if debug:
        typhoonname = 'CHANTHU'
        remote_dir = '20210910120000'
        logger.info(f"DEBUGGING piepline for typhoon{typhoonname}")  
    fc = Forecast(main_path,remote_dir,typhoonname, countryCodeISO3='PHP', admin_level=3)
    #fc.data_filenames_list
    #fc.image_filenames_list
    landfall_time='NA'
    landfall_location_manucipality='NA'
    EAP_TRIGGERED='no'
    landfall_time_hr='NA-hour'
    IBF_API_URL = fc.API_SERVICE_URL
    ADMIN_LOGIN = fc.ADMIN_LOGIN
    ADMIN_PASSWORD = fc.ADMIN_PASSWORD

    # login
    login_response = requests.post(f'{IBF_API_URL}/api/user/login',
                                   data=[('email', ADMIN_LOGIN), ('password', ADMIN_PASSWORD)])
    if login_response.status_code >= 400:
        logging.error(f"PIPELINE ERROR AT LOGIN {login_response.status_code}: {login_response.text}")
        sys.exit()
    token = login_response.json()['user']['token']
    
    if not fc.Activetyphoon: #if it is not empty   
        for typhoon_names in fc.Activetyphoon:
            landfall_location=fc.landfall_location[typhoon_names]
            if landfall_location:#if dict is not empty         
                landfall_time=list(landfall_location.items())[0][0] #'YYYYMMDDHH'
                landfall_time_obj = datetime.strptime(landfall_time, '%Y%m%d%H%M')
                landfall_dellta = (start_time - landfall_time_obj)#.strftime("%Y%m%d")
                seconds = landfall_dellta.total_seconds()
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                seconds = seconds % 60
                landfall_time_hr= str(hours)+'-hour'                
                landfall_location_manucipality=list(landfall_location.items())[0][1]['adm3_pcode'] #pcode?         
            
            #adm3_pcode,storm_id,is_ensamble,value_count,v_max,name,dis_track_min
            typhoon_windAll=fc.typhhon_wind_data[typhoon_names]
            typhoon_windAll=typhoon_windAll.query('is_ensamble=="False"')
            typhoon_windAll['alert_threshold']=typhoon_windAll['dis_track_min'].apply(lambda x:1 if x< 100 else 0)
            
            typhoon_wind=typhoon_windAll[['adm3_pcode','v_max','dis_track_min','alert_threshold']]
            typhoon_wind.rename(columns={"v_max": "windspeed"},inplace=True) 
            #max_06h_rain,max_24h_rain,Mun_Code
            typhoon_rainfall=fc.rainfall_data[typhoon_names][['Mun_Code','max_24h_rain']]
            typhoon_rainfall.rename(columns={"Mun_Code": "adm3_pcode","max_24h_rain": "rainfall"},inplace=True) 
            df_hazard=pd.merge(typhoon_windAll, typhoon_rainfall,  how='left', left_on='adm3_pcode',right_on = 'adm3_pcode')
            #"","adm3_en","glat","adm3_pcode","adm2_pcode","adm1_pcode","glon","GEN_mun_code","probability_dist50","impact","WEA_dist_track"
            with open (fc.Output_folder+"Average_Impact_"+fc.date_dir+"_"+typhoon_names+".csv") as csv_file2:
                impact=pd.read_csv(csv_file2)
            impact_df=impact[["adm3_pcode","probability_dist50","impact"]]
            impact_df.rename(columns={"adm3_pcode": "placeCode","impact": "house_affected","probability_dist50": "prob_within_50km"},inplace=True)  
            df_total=pd.merge(impact_df, df_hazard,  how='left', left_on='adm3_pcode',right_on = 'adm3_pcode')

            #"","Typhoon_name",">=100k",">=80k",">=70k",">=50k",">=30k","trigger"
            
            with open (fc.Output_folder+"trigger_"+fc.date_dir+"_"+typhoon_names+".csv") as csv_file1:
                df=pd.read_csv(csv_file1)
                for index, row in df.iterrows():
                    trigger = int(row['trigger'])            
                    if trigger==1:
                        EAP_TRIGGERED='yes'
                        EAP_TRIGGERED_bool='true'
                        
            trigger_per_day_PHP=[{"landfall_location": landfall_location_manucipality,
            "landfall_time": landfall_time,"lead_time": landfall_time_hr,
            "EAP_triggered": EAP_TRIGGERED}]             
            
            typhoon_track=fc.hrs_track_data[typhoon_names]
            typhoon_track['timestampOfTrackpoint'] = pd.to_datetime(typhoon_track['YYYYMMDDHH'], format='%Y%m%d%H%M').dt.strftime('%m-%d-%Y %H:%M:%S')
            typhoon_track.rename(columns={"LON": "lon","LAT": "lat"},inplace=True) 
            wind_track =typhoon_track[['lon','lat','timestampOfTrackpoint']]
            
            ##############################################################################    
            #upload track 
            
            exposure_data = {'countryCodeISO3': 'PHL'}
            exposure_data["leadTime"] = landfall_time_hr
            exposure_data["eventName"] = typhoon_names
            exposure_place_codes = []
            
            for ix, row in wind_track.iterrows():
                exposure_entry = {'lat': row['lat'],
                                  'lon': row['lon'],
                                  'timestampOfTrackpoint': row['timestampOfTrackpoint']}
                exposure_place_codes.append(exposure_entry)
            exposure_data['trackpointDetails'] = exposure_place_codes


            # upload data
            upload_response = requests.post(f'{IBF_API_URL}/api/tphoon-track',
                                            json=exposure_data,
                                            headers={'Authorization': 'Bearer '+token,
                                                     'Content-Type': 'application/json',
                                                     'Accept': 'application/json'})
            print(upload_response)
            print(layer)  
            if upload_response.status_code >= 400:
                logging.error(f"PIPELINE ERROR AT UPLOAD {login_response.status_code}: {login_response.text}")  
                
            ##############################################################################
            #upload event 
            exposure_data = {'countryCodeISO3': 'PHL'}
            exposure_data["disasterType"] = 'typhoon'
            exposure_data["eventName"] = typhoon_names 
            exposure_place_codes = []
            exposure_entry = {'leadTime': landfall_time_hr,
                              'triggered': EAP_TRIGGERED_bool}
            exposure_place_codes.append(exposure_entry)
      
            exposure_data['triggersPerLeadTime'] = exposure_place_codes


            # upload data
            upload_response = requests.post(f'{IBF_API_URL}/api/event/triggers-per-leadtime',
                                            json=exposure_data,
                                            headers={'Authorization': 'Bearer '+token,
                                                     'Content-Type': 'application/json',
                                                     'Accept': 'application/json'})
            print(upload_response)
            print(layer)
            if upload_response.status_code >= 400:
                logging.error(f"PIPELINE ERROR AT UPLOAD {login_response.status_code}: {login_response.text}")            
            
            ##############################################################################            
            # upload dynamic layers

            for layer in ["windspeed","alert_threshold","rainfall","adm3_pcode","prob_within_50km","houses_affected"]:

                # prepare layer
                #exposure_data = {'countryCodeISO3': countrycode}
                exposure_data = {'countryCodeISO3': 'PHL'}
                exposure_place_codes = []
                for ix, row in df_total.iterrows():
                    exposure_entry = {'placeCode': row['adm3_pcode'],
                                      'amount': row[layer]}
                    exposure_place_codes.append(exposure_entry)
                    exposure_data['exposurePlaceCodes'] = exposure_place_codes
                    exposure_data["adminLevel"] = 3
                    exposure_data["leadTime"] = landfall_time_hr
                    exposure_data["dynamicIndicator"] = layer
                    exposure_data["disasterType"] = 'typhoon'
                    exposure_data["eventName"] = typhoon_names 

                    # upload data
                    upload_response = requests.post(f'{IBF_API_URL}/api/admin-area-dynamic-data/exposure',
                                                    json=exposure_data,
                                                    headers={'Authorization': 'Bearer '+token,
                                                             'Content-Type': 'application/json',
                                                             'Accept': 'application/json'})
                    print(lead_time)
                    print(upload_response)
                    print(layer)
                    if upload_response.status_code >= 400:
                        logging.error(f"PIPELINE ERROR AT UPLOAD {login_response.status_code}: {login_response.text}")
                        #sys.exit()            


            # typhoon_wind.rename(columns={"adm3_pcode": "placeCode","v_max": "amount"},inplace=True) 
            # out = typhoon_wind.to_json(orient='records')
            
            # exposure_data = {"countryCodeISO3": "PHL"}  
            # exposure_data["exposurePlaceCodes"] = out
            # exposure_data["adminLevel"] = 3 
            # exposure_data["leadTime"] = landfall_time_hr 
            # exposure_data["dynamicIndicator"] = "f" 
            # exposure_data["disasterType"] = "typhoon"
            # exposure_data["eventName"] = typhoon_names               
            
            # with open(main_path+f"forecast/triggers/windspeed_{typhoon_names}.json", 'w') as fp:
                # fp.write(exposure_data)
            # typhoon_wind=typhoon_windAll[['adm3_pcode','dis_track_min']]
            # typhoon_wind['alert']=typhoon_wind['dis_track_min'].apply(lambda x:'Yes' if x< 100 else 'No')
            
            # typhoon_wind.rename(columns={"adm3_pcode": "placeCode","alert": "amount"},inplace=True) 
            # out = typhoon_wind.to_json(orient='records')
            # exposure_data = {"countryCodeISO3": "PHL"}  
            # exposure_data["exposurePlaceCodes"] = out
            # exposure_data["adminLevel"] = 3 
            # exposure_data["leadTime"] = landfall_time_hr 
            # exposure_data["dynamicIndicator"] = "alert_threshold" 
            # exposure_data["disasterType"] = "typhoon"
            # exposure_data["eventName"] = typhoon_names
            
            # with open(main_path+f"forecast/triggers/alert_threshold_{typhoon_names}.json", 'w') as fp:
                # fp.write(exposure_data)
            # #max_06h_rain,max_24h_rain,Mun_Code
            # typhoon_rainfall=fc.rainfall_data[typhoon_names][['Mun_Code','max_24h_rain']]
            # typhoon_rainfall.rename(columns={"Mun_Code": "placeCode","max_24h_rain": "amount"},inplace=True)   
            # out = typhoon_rainfall.to_json(orient='records')
            # exposure_data = {"countryCodeISO3": "PHL"}  
            # exposure_data["exposurePlaceCodes"] = out
            # exposure_data["adminLevel"] = 3 
            # exposure_data["leadTime"] = landfall_time_hr 
            # exposure_data["dynamicIndicator"] = "rainfall" 
            # exposure_data["disasterType"] = "typhoon"
            # exposure_data["eventName"] = typhoon_names
            # with open(main_path+f"forecast/triggers/rainfall_{typhoon_names}.json", 'w') as fp:
                # fp.write(exposure_data)                
                        
            # #"","adm3_en","glat","adm3_pcode","adm2_pcode","adm1_pcode","glon","GEN_mun_code","probability_dist50","impact","WEA_dist_track"
            # with open (fc.Output_folder+"Average_Impact_"+fc.date_dir+"_"+typhoon_names+".csv") as csv_file2:
                # impact=pd.read_csv(csv_file2)
            # impact_df=impact["adm3_pcode","impact"]
            # impact_df.rename(columns={"adm3_pcode": "placeCode","impact": "amount"},inplace=True)            
            
            # out = impact_df.to_json(orient='records')
            
            # exposure_data = {"countryCodeISO3": "PHL"}  
            # exposure_data["exposurePlaceCodes"] = out
            # exposure_data["adminLevel"] = 3 
            # exposure_data["leadTime"] = landfall_time_hr 
            # exposure_data["dynamicIndicator"] = "house_affected" 
            # exposure_data["disasterType"] = "typhoon"
            # exposure_data["eventName"] = typhoon_names
            
            # with open(main_path+f"forecast/triggers/house_affected_{typhoon_names}.json", 'w') as fp:
                # fp.write(exposure_data)
            # prob_50_df=impact["adm3_pcode","probability_dist50"]
            # prob_50_df.rename(columns={"adm3_pcode": "placeCode","probability_dist50": "amount"},inplace=True)            
            # out = prob_50_df.to_json(orient='records')
            
            # exposure_data = {"countryCodeISO3": "PHL"}  
            # exposure_data["exposurePlaceCodes"] = out
            # exposure_data["adminLevel"] = 3 
            # exposure_data["leadTime"] = landfall_time_hr 
            # exposure_data["dynamicIndicator"] = "prob_within_50km" 
            # exposure_data["disasterType"] = "typhoon"
            # exposure_data["eventName"] = typhoon_names
            
            # with open(main_path+f"forecast/triggers/prob_within_50km_{typhoon_names}.json", 'w') as fp:
                # fp.write(exposure_data)
                

 


    print('---------------------AUTOMATION SCRIPT FINISHED---------------------------------')
    print(str(datetime.now()))


#%%#Download rainfall (old pipeline)
#automation_sript(path)
if __name__ == "__main__":
    main()

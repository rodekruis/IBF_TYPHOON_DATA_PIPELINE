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
from shapely.geometry import Point, Polygon, MultiPolygon, box

from typhoonmodel.utility_fun import track_data_clean, Check_for_active_typhoon, Sendemail, \
    ucl_data, plot_intensity, initialize

if platform == "linux" or platform == "linux2": #check if running on linux or windows os
    from typhoonmodel.utility_fun import Rainfall_data
elif platform == "win32":
    from typhoonmodel.utility_fun import Rainfall_data_window as Rainfall_data
from typhoonmodel.utility_fun.forecast_process import Forecast
decoder = Decoder()
import requests

from fiona.crs import from_epsg
from climada.util import coordinates  

 
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
    print('---------------------AUTOMATION SCRIPT STARTED---------------------------------')
    print(str(start_time))
    #%% check for active typhoons
    print('---------------------check for active typhoons---------------------------------')
    print(str(start_time))
    remote_dir = remote_directory
    main_path=path
    
    ###############################################################
    ####  DEBUG flag acts like a mock data 
    if debug:
        typhoonname = 'CHANTHU'
        remote_dir = '20210910120000'
        logger.info(f"DEBUGGING piepline for typhoon{typhoonname}")  
        
 
        
    fc = Forecast(main_path,remote_dir,typhoonname, countryCodeISO3='PHP', admin_level=3)
 
    landfall_time='NA'
    landfall_location_manucipality='NA'
    EAP_TRIGGERED_bool=0
    EAP_TRIGGERED='no'
    ################################################## if there is no Landfall what should be the number for leadtime
    #################################################   the api accept only numbers and NA is not an option 
    landfall_time_hr='99-hour'
    IBF_API_URL = 'https://ibf-test.510.global/api/' #fc.API_SERVICE_URL
    ADMIN_LOGIN = 'dunant@redcross.nl'#fc.ADMIN_LOGIN
    ADMIN_PASSWORD ='password'# fc.ADMIN_PASSWORD
    
    # login
    login_response = requests.post(fc.API_SERVICE_URL  +'user/login',
                                   data=[('email', ADMIN_LOGIN), ('password', ADMIN_PASSWORD)])
                                   
    if login_response.status_code >= 400:
        logging.error(f"PIPELINE ERROR AT LOGIN {login_response.status_code}: {login_response.text}")
        sys.exit()
        
    token = login_response.json()['user']['token']
    rainfall_=pd.read_csv(os.path.join(fc.Input_folder, "rainfall/rain_data.csv"))
  
    
    if fc.Activetyphoon: #if it is not empty   
        for typhoon_names in fc.Activetyphoon:
            ####### landfall location is a dict {"typhoon name":{"landfall  time":" ","landfall location ":" "}}
            landfall_location=fc.landfall_location[typhoon_names]
            fname=open(os.path.join(fc.main_path,'forecast/Input/',"typhoon_info_for_model.csv"),'w')
            fname.write('source,filename,event,time'+'\n')   
            line_='Rainfall,'+'%srainfall' % fc.Input_folder +',' +typhoon_names+','+ fc.date_dir  
            fname.write(line_+'\n')
            line_='Output_folder,'+'%s' % fc.Output_folder +',' +typhoon_names+',' + fc.date_dir  
            fname.write(line_+'\n')
            fc.hrs_track_data[typhoon_names].to_csv(os.path.join(fc.Input_folder,'ecmwf_hrs_track.csv'), index=False)
            line_='ecmwf,'+'%secmwf_hrs_track.csv' % fc.Input_folder+ ',' +typhoon_names+','+ fc.date_dir   
            fname.write(line_+'\n') 
            fc.typhhon_wind_data[typhoon_names].to_csv(os.path.join(fc.Input_folder,'windfield.csv'), index=False)
            line_='windfield,'+'%swindfield.csv' % fc.Input_folder+ ',' +typhoon_names+','+ fc.date_dir   #StormName #
            fname.write(line_+'\n')
            fname.close()
            
            hrs_track_df=fc.hrs_track_data[typhoon_names]
            forecast_time=str(hrs_track_df.index[0])
            forecast_time = datetime.strptime(forecast_time, '%Y-%m-%d %H:%M:%S') 
            
            ### filter areas in Philippiness area of responsibility 
            hrs_track_df=hrs_track_df.query('5 < LAT < 20 and 115 < LON < 133')
            coord_lat = gpd.GeoDataFrame(hrs_track_df.VMAX, geometry=gpd.points_from_xy(hrs_track_df.LON,hrs_track_df.LAT))
            coord_lat.set_crs(epsg=4326, inplace=True)
            DIST_TO_COAST_M=coordinates.dist_to_coast(coord_lat, lon=None, signed=False)

            hrs_track_df['distance']=DIST_TO_COAST_M
            hrs_track_df.reset_index(inplace=True)
             
            if (hrs_track_df['distance'].values < 0).any():
                for row,data in hrs_track_df.iterrows():
                    landfalltime=data['time']
                    landfalltime_time_obj =datetime.strptime(str(landfalltime), '%Y-%m-%d %H:%M:%S')
                    if data['distance'] < 0:     
                        break
            else:
                landfalltime=hrs_track_df.sort_values(by='distance',ascending=True).iloc[0]['time']
                landfalltime_time_obj = datetime.strptime(str(landfalltime), '%Y-%m-%d %H:%M:%S') 
            landfall_dellta = ( landfalltime_time_obj-forecast_time)#.strftime("%Y%m%d")
            seconds = landfall_dellta.total_seconds()
            hours = int(seconds // 3600)
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            landfall_time_hr= str(hours)+'-hour'  
            
            #############################################################
            os.chdir(fc.main_path)
            
            if platform == "linux" or platform == "linux2": #check if running on linux or windows os
                # linux
                try:
                    p = subprocess.check_call(["Rscript", "run_model_V2.R", str(fc.rainfall_error)])
                except subprocess.CalledProcessError as e:
                    logger.error(f'failed to excute R sript')
                    raise ValueError(str(e))
            elif platform == "win32": #if OS is windows edit the path for Rscript
                try:
                    p = subprocess.check_call(["C:/Program Files/R/R-4.1.0/bin/Rscript", "run_model_V2.R", str(fc.rainfall_error)])
                except subprocess.CalledProcessError as e:
                    logger.error(f'failed to excute R sript')
                    raise ValueError(str(e))            

            #if landfall_location:#if dict is not empty         
                #landfall_time=list(landfall_location.items())[0][0] #'YYYYMMDDHH'
                #landfall_time_obj = datetime.strptime(landfall_time, '%Y%m%d%H%M')
                #landfall_dellta = (start_time - landfall_time_obj)#.strftime("%Y%m%d")
                #seconds = landfall_dellta.total_seconds()
                #hours = seconds // 3600
                #minutes = (seconds % 3600) // 60
                #seconds = seconds % 60
                #landfall_time_hr= str(hours)+'-hour'                
                #landfall_location_manucipality=list(landfall_location.items())[0][1]['adm3_pcode'] #pcode?         
            
            #adm3_pcode,storm_id,is_ensamble,value_count,v_max,name,dis_track_min
            typhoon_windAll=fc.typhhon_wind_data[typhoon_names]
            typhoon_windAll=typhoon_windAll.query('is_ensamble=="False"')
            
            
          
            typhoon_wind=typhoon_windAll[['adm3_pcode','v_max','dis_track_min']].drop_duplicates('adm3_pcode')
            typhoon_wind.rename(columns={"v_max": "windspeed"},inplace=True) 
            logger.info(f"{len(typhoon_wind)}")
            #max_06h_rain,max_24h_rain,Mun_Code
            typhoon_rainfall=rainfall_[['Mun_Code','max_24h_rain']]
            typhoon_rainfall.rename(columns={"Mun_Code": "adm3_pcode","max_24h_rain": "rainfall"},inplace=True)
            logger.info(f"{len(typhoon_rainfall)}")
            
            #create dataframe for all manucipalities 
            admin_df=fc.pcode 
            df_wind=pd.merge(admin_df,typhoon_wind,  how='left', left_on='adm3_pcode',right_on = 'adm3_pcode')            
            df_hazard=pd.merge(df_wind, typhoon_rainfall,  how='left', left_on='adm3_pcode',right_on = 'adm3_pcode')
            #df_hazard=df_hazard.fillna(0)
            #"","adm3_en","glat","adm3_pcode","adm2_pcode","adm1_pcode","glon","GEN_mun_code","probability_dist50","impact","WEA_dist_track"
            with open (fc.Output_folder+"Average_Impact_"+fc.date_dir+"_"+typhoon_names+".csv") as csv_file2:
                impact=pd.read_csv(csv_file2)
            impact_df=impact[["adm3_pcode","probability_dist50","impact","WEA_dist_track"]]
            impact_df.rename(columns={"impact": "houses_affected","probability_dist50": "prob_within_50km"},inplace=True)  
            logger.info(f"{len(impact_df)}")
            df_total=pd.merge(df_hazard,impact_df,  how='left', left_on='adm3_pcode',right_on = 'adm3_pcode')
            df_total['show_admin_area']=df_total['WEA_dist_track'].apply(lambda x:1 if x< 100 else 0)
            df_total['alert_threshold']=df_total['houses_affected'].apply(lambda x:1 if x< 5 else 0)
            df_total=df_total.fillna(0)
            df_total=df_total.drop_duplicates('adm3_pcode')
            df_total['alert_threshold']=0
            logger.info(f"{len(df_total)}")

            #"","Typhoon_name",">=100k",">=80k",">=70k",">=50k",">=30k","trigger"
            
            with open (fc.Output_folder+"trigger_"+typhoon_names+".csv") as csv_file1:
                df=pd.read_csv(csv_file1)
                for index, row in df.iterrows():
                    trigger = int(row['trigger'])            
                    if trigger==1:
                        EAP_TRIGGERED='yes'
                        EAP_TRIGGERED_bool=1
                        df_total['alert_threshold']=1 
            
            trigger_per_day_PHP=[{#"landfall_location": landfall_location_manucipality,
            "landfall_time": landfalltime_time_obj,"lead_time": landfall_time_hr,
            "EAP_triggered": EAP_TRIGGERED}]             
            
            typhoon_track=fc.hrs_track_data[typhoon_names]
            typhoon_track['timestampOfTrackpoint'] = pd.to_datetime(typhoon_track['YYYYMMDDHH'], format='%Y%m%d%H%M').dt.strftime('%m-%d-%Y %H:%M:%S')
            typhoon_track.rename(columns={"LON": "lon","LAT": "lat"},inplace=True) 
            wind_track =typhoon_track[['lon','lat','timestampOfTrackpoint']]
            
            ##############################################################################    
            #upload track 
            
            exposure_data = {"countryCodeISO3": "PHL"}
            exposure_data["leadTime"] = landfall_time_hr
            exposure_data["eventName"] = typhoon_names
            exposure_place_codes = []
            
            for ix, row in wind_track.iterrows():
                exposure_entry = {"lat": row["lat"],
                                  "lon": row["lon"],
                                  "timestampOfTrackpoint": row["timestampOfTrackpoint"]}
                exposure_place_codes.append(exposure_entry)
            exposure_data["trackpointDetails"] = exposure_place_codes


            # upload data
            upload_response = requests.post(fc.API_SERVICE_URL + 'typhoon-track',
                                            json=exposure_data,
                                            headers={'Authorization': 'Bearer '+token,
                                                     'Content-Type': 'application/json',
                                                     'Accept': 'application/json'})
            print(upload_response)
            logger.info(f"track upload {upload_response.status_code}")
 
            if upload_response.status_code >= 400:
                logging.error(f"PIPELINE ERROR AT UPLOAD {login_response.status_code}: {login_response.text}")  
                
            ##############################################################################
            # #upload event 
            # exposure_data = {'countryCodeISO3': 'PHL'}
            # exposure_data["disasterType"] = 'typhoon'
            # exposure_data["eventName"] = typhoon_names 
            # exposure_place_codes = []
            # exposure_entry = {'leadTime': landfall_time_hr,
                              # 'triggered': EAP_TRIGGERED_bool}
            # exposure_place_codes.append(exposure_entry)
      
            # exposure_data['triggersPerLeadTime'] = exposure_place_codes


            # upload data
            # upload_response = requests.post(f'{IBF_API_URL}/api/event/triggers-per-leadtime',
                                            # json=exposure_data,
                                            # headers={'Authorization': 'Bearer '+token,
                                                     # 'Content-Type': 'application/json',
                                                     # 'Accept': 'application/json'})
            # print(upload_response)
            # print(layer)
            # if upload_response.status_code >= 400:
                # logging.error(f"PIPELINE ERROR AT UPLOAD {login_response.status_code}: {login_response.text}")            
            
            ##############################################################################            
            # upload dynamic layers

            for layer in ["windspeed","rainfall", "prob_within_50km","houses_affected","alert_threshold","show_admin_area"]:

                # prepare layer
                logger.info(f"preparing data for {layer}")
                #exposure_data = {'countryCodeISO3': countrycode}
                exposure_data = {"countryCodeISO3": "PHL"}
                exposure_place_codes = []
                #### change the data frame here to include impact
                for ix, row in df_total.iterrows():
                    exposure_entry = {"placeCode": row["adm3_pcode"],
                                      "amount": row[layer]}
                    exposure_place_codes.append(exposure_entry)
                exposure_data["exposurePlaceCodes"] = exposure_place_codes
                exposure_data["adminLevel"] = 3
                exposure_data["leadTime"] = landfall_time_hr
                exposure_data["dynamicIndicator"] = layer
                exposure_data["disasterType"] = "typhoon"
                exposure_data["eventName"] = typhoon_names 

                # upload data
                upload_response = requests.post(fc.API_SERVICE_URL + 'admin-area-dynamic-data/exposure',
                                                json=exposure_data,
                                                headers={'Authorization': 'Bearer '+token,
                                                         'Content-Type': 'application/json',
                                                         'Accept': 'application/json'}) 
                print(upload_response)
                print(layer)
                logging.info(f"uploading layers {layer}")
                logger.info(f"layer upload report {upload_response.status_code}")
                if upload_response.status_code >= 400:
                    logging.error(f"PIPELINE ERROR AT UPLOAD {login_response.status_code}: {login_response.text}")
                    #sys.exit()            


    #if there is no active typhoon 
    else: #if it is not empty 
        admin_df=fc.pcode
        
        admin_df['show_admin_area']=0
        admin_df['alert_threshold']=0
        exposure_entry=[]
        exposure_data = {"countryCodeISO3": "PHL"}
        for ix, row in admin_df.iterrows():
            exposure_entry = {"placeCode": row["adm3_pcode"],
                              "amount": row['alert_threshold']}
            exposure_place_codes.append(exposure_entry)
            
        exposure_data["exposurePlaceCodes"] = exposure_place_codes
        exposure_data["adminLevel"] = 3
        exposure_data["leadTime"] = landfall_time_hr
        exposure_data["dynamicIndicator"] = 'alert_threshold'
        exposure_data["disasterType"] = "typhoon"
        exposure_data["eventName"] = 'NA' 
        log.info('no active Typhoon')
        

    print('---------------------AUTOMATION SCRIPT FINISHED---------------------------------')
    print(str(datetime.now()))


#%%#Download rainfall (old pipeline)
#automation_sript(path)
if __name__ == "__main__":
    main()

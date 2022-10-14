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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
from pybufrkit.decoder import Decoder
import numpy as np
from geopandas.tools import sjoin
import geopandas as gpd
import click
import json
from shapely import wkb, wkt
from shapely.geometry import Point, Polygon
from pathlib import Path
from climada.hazard import Centroids, TropCyclone, TCTracks
from climada.hazard.tc_tracks_forecast import TCForecast
from typhoonmodel.utility_fun.settings import *
from typhoonmodel.utility_fun.dynamicDataDb import DatabaseManager
from climada.util import coordinates 
from typhoonmodel.utility_fun import (
    track_data_clean,
    Check_for_active_typhoon,
    Sendemail,
    #ucl_data,
    plot_intensity,
    initialize,
)


if (
    platform == "linux" or platform == "linux2"
):  # check if running on linux or windows os
    from typhoonmodel.utility_fun import Rainfall_data
elif platform == "win32":
    from typhoonmodel.utility_fun import Rainfall_data_window as Rainfall_data
decoder = Decoder()
initialize.setup_logger()
logger = logging.getLogger(__name__)




class Forecast:
    def __init__(self, remote_dir, countryCodeISO3, admin_level):
        self.db = DatabaseManager(countryCodeISO3, admin_level)
        self.TyphoonName = typhoon_event_name
        self.admin_level = admin_level
        self.remote_dir = remote_dir
 
        self.Wind_damage_radius=Wind_damage_radius
        self.Population_Growth_factor=Population_Growth_factor #(1+0.02)^7 adust 2015 census data by 2%growth for the pst 7 years 
        self.ECMWF_MAX_TRIES = 3
        self.ECMWF_SLEEP = 30  # s
        self.main_path = MAIN_DIRECTORY
        self.Show_Areas_on_IBF_radius=Show_Areas_on_IBF_radius
        ##Create grid points to calculate Winfield
        cent = Centroids()
        cent.set_raster_from_pnt_bounds((118, 6, 127, 19), res=0.05)
        cent.check()
        #cent.plot()
        self.cent=cent
        self.ECMWF_CORRECTION_FACTOR=ECMWF_CORRECTION_FACTOR
        self.ECMWF_LATENCY_LEADTIME_CORRECTION=ECMWF_LATENCY_LEADTIME_CORRECTION

        admin = gpd.read_file(
            ADMIN_PATH
        )  # gpd.read_file(os.path.join(self.main_path,"data/gis_data/phl_admin3_simpl2.geojson"))
        pcode = pd.read_csv(
            os.path.join(self.main_path, "data/pre_disaster_indicators/pcode.csv")
        )
        
        Tphoon_EAP_Areas = pd.read_csv(
            os.path.join(self.main_path, "data/Tphoon_EAP_Areas.csv")
            )
        
        self.Tphoon_EAP_Areas = Tphoon_EAP_Areas
        self.pre_disaster_inds = self.pre_disaster_data()
        self.pcode = pcode
        
        df = pd.DataFrame(data=cent.coord) 
        df["centroid_id"] = "id" + (df.index).astype(str)
        
        centroid_idx = df["centroid_id"].values
        
        ## sometimes there is a problem to correctly generate datafram from centroids 
        ## temporary fix for this is to read the datafram from file 
        
        if len(centroid_idx)==0:
            df = gpd.read_file(CENTROIDS_PATH)
            centroid_idx = df["centroid_id"].values  
            
        self.centroid_idx=centroid_idx
        self.ncents = cent.size
   
        df = df.rename(columns={0: "lat", 1: "lon"})
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        admin.set_crs(epsg=4326, inplace=True)
        df.set_crs(epsg=4326, inplace=True)
        
        self.dfGrids=df

        df_admin = sjoin(df, admin, how="left").dropna()
        self.df_admin = df_admin
        self.admin = admin

        # Sometimes the ECMWF ftp server complains about too many requests
        # This code allows several retries with some sleep time in between
        n_tries = 0

        while True:
            try:
                logger.info("Downloading ECMWF typhoon tracks")
                bufr_files = TCForecast.fetch_bufr_ftp(remote_dir=self.remote_dir)
                fcast = TCForecast()
                fcast.fetch_ecmwf(files=bufr_files)
            except ftplib.all_errors as e:
                n_tries += 1
                if n_tries >= self.ECMWF_MAX_TRIES:
                    logger.error(
                        f" Data downloading from ECMWF failed: {e}, "
                        f"reached limit of {self.ECMWF_MAX_TRIES} tries, exiting"
                    )
                    sys.exit()
                logger.error(
                    f" Data downloading from ECMWF failed: {e}, retrying after {self.ECMWF_SLEEP} s"
                )
                time.sleep(self.ECMWF_SLEEP)
                continue
            break

        """
        filter data downloaded in the above step for active typhoons  in PAR
        https://bagong.pagasa.dost.gov.ph/learning-tools/philippine-area-of-responsibility
        : 4°N 114°E, 28°N 114°E, 28°N 145°E and 4°N 145°N
        """
        
        FcastData = [tr for tr in fcast.data if (tr.basin =="W - North West Pacific")]

        TropicalCycloneAdvisoryDomain_events = list(
            set(
                [
                    tr.name
                    for tr in FcastData
                    if ( 
                        sum(~np.isnan(tr.max_sustained_wind.values))> sum(np.isnan(tr.max_sustained_wind.values)) 
                        and np.nanmin(tr.lat.values) < 21
                        and np.nanmax(tr.lat.values) > 5
                        and np.nanmin(tr.lon.values) < 130
                        and np.nanmax(tr.lon.values) > 116
                        and tr.is_ensemble == "False"
                    )
                ]
            )
        )
        
        self.TropicalCycloneAdvisoryDomain_events=TropicalCycloneAdvisoryDomain_events
        
        #FcastData = [tr for tr in FcastData if (tr.is_ensemble == "False")] limit running in azure logic
        if High_resoluation_only_Switch:
            Data_to_process = [tr for tr in FcastData if (tr.is_ensemble == "False")]
        else:
            Data_to_process =FcastData        
        

        fcast_data = [
            track_data_clean.track_data_clean(tr)
            for tr in Data_to_process
            if (tr.time.size > 1 and tr.name in TropicalCycloneAdvisoryDomain_events)
        ]

        self.fcast_data = fcast_data
        #self.forecast_time = fcast_data[0].forecast_time.strftime("%Y%m%d%H") 
        self.ecmwf = FcastData
        #fcast.data = fcast_data  # [tr for tr in fcast.data if tr.name in Activetyphoon]
        self.data_filenames_list = {}
        self.image_filenames_list = {}
        self.typhoon_wind_data = {}
        self.eap_status = {}
        self.eap_status_bool = {}
        self.hrs_track_data = {}
        self.landfall_location = {}
        self.Activetyphoon  = []  # Activetyphoon
        self.Activetyphoon_landfall = {}    # Activetyphoon
        self.Input_folder = Input_folder
        self.rainfall_path = rainfall_path
        self.Output_folder = Output_folder

        self.WIND_SPEED_THRESHOLD =  WIND_SPEED_THRESHOLD  #
        
        if TropicalCycloneAdvisoryDomain_events != []:
            # download NOAA rainfall
            try:
                Rainfall_data.download_rainfall_nomads()

                rainfall_data = pd.read_csv(
                    os.path.join(self.rainfall_path, "rain_data.csv")
                )
                rainfall_data.rename(
                    columns={
                        "max_06h_rain": "HAZ_max_06h_rain",
                        "Mun_Code":"adm3_pcode",
                        "max_24h_rain": "HAZ_rainfall_max_24h",
                    },
                    inplace=True,
                )
                self.rainfall_data = rainfall_data
                rainfall_error = False
            except:
                traceback.print_exc()
                logger.info(
                    f"Rainfall download failed, performing download in R script"
                )
                rainfall_error = True

            self.rainfall_error = rainfall_error
            
            if  Active_Typhoon_event_list:
                Active_Typhoon_events=Active_Typhoon_event_list
            else:
                Active_Typhoon_events=TropicalCycloneAdvisoryDomain_events
     
            for typhoons in Active_Typhoon_events:
                #
                logger.info(f"Processing data {typhoons}")
                
                
 

 
                HRS = [ tr  for tr in fcast_data  if (tr.is_ensemble=='False' and tr.name in [typhoons]) ]
                
                self.Wind_damage_radius=np.nanmax(HRS[0].max_radius.values)
                self.forecast_time = fcast_data[0].forecast_time.strftime("%Y%m%d%H") 
                
                
                ### run wind field function 
                
                is_land_fall=self.windfieldData(typhoons) 
                logger.info('finished wind field calculation')
          
                if  is_land_fall ==0:#not bool(is_land_fall):                
                    #check if calculated wind fields are empty 
                    logger.info(f'{typhoons}event didnt made landfall yet')
                    self.Activetyphoon_landfall[typhoons]='notmadelandfall'       
                elif is_land_fall ==1:
                    logger.info(f'there isa lready a landfall event {typhoons}')
                    self.Activetyphoon_landfall[typhoons]='madelandfall' 
                elif is_land_fall ==2:
                    self.Activetyphoon_landfall[typhoons]='Farfromland'
                    json_path = self.Output_folder  + typhoons  
                    self.db.uploadTyphoonDataNoLandfall(json_path)
                    self.db.uploadTrackData(json_path)
                    
                
                              
                wind_file_path=os.path.join(self.Input_folder, f"{typhoons}_windfield.csv")  
                              
                if os.path.isfile(wind_file_path):
                    calcuated_wind_fields=pd.read_csv(wind_file_path)                
                    if not calcuated_wind_fields.empty:
                        self.Activetyphoon.append(typhoons)
                        self.impact_model(typhoon_names=typhoons,wind_data=calcuated_wind_fields)  
                        logger.info('go to data upload ')                          
                        '''
                        forecast_directory=typhoons+fcast_data[0].forecast_time.strftime("%Y%m%d%H%M") 
                        try:
                            logger.info('upload result to datalake')                          
                            self.db.postDataToDatalake(datalakefolder=forecast_directory)
                        except:
                            logger.info('failed to upload result to datalake')   
                            pass
                        '''

                logger.info(f'go to data upload {self.Activetyphoon_landfall}')   
                
    def min_distance(self,point, lines):
        return lines.distance(point).min()

    def model(self, df_total):
        from sklearn.model_selection import (
            GridSearchCV,
            RandomizedSearchCV,
            StratifiedKFold,
            train_test_split,
            KFold)
        from sklearn.metrics import (
            mean_absolute_error, 
            mean_squared_error,
            recall_score,
            f1_score,
            precision_score,
            confusion_matrix,
            make_scorer)
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        from xgboost.sklearn import XGBRegressor
        import xgboost as xgb


        combined_input_data = pd.read_csv(ML_model_input)
        tphoon_events = (
            combined_input_data[["typhoon", "DAM_perc_dmg"]]
            .groupby("typhoon")
            .size()
            .to_dict()
        )
        combined_input_data['year']=combined_input_data['typhoon'].apply(lambda x:x[-4:])
        Housing_unit_correction_df=pd.DataFrame.from_dict(Housing_unit_correction)
        combined_input_data = pd.merge(combined_input_data, Housing_unit_correction_df,  how='left', left_on='year', right_on ='year')
        combined_input_data["DAM_perc_dmg"] = combined_input_data[
            ["HAZ_v_max", "HAZ_rainfall_Total", "DAM_perc_dmg","facor"]
        ].apply(self.set_zeros, axis="columns")


        selected_features_xgb_regr = [
            "HAZ_v_max",
            #"HAZ_rainfall_max_24h",            
            "HAZ_dis_track_min",
            "TOP_mean_slope",
            "TOP_mean_elevation_m",
            "TOP_ruggedness_stdev",
            "TOP_mean_ruggedness",
            "TOP_slope_stdev",
            "VUL_poverty_perc",
            "GEN_with_coast",
            "VUL_Housing_Units",
            "VUL_StrongRoof_StrongWall",
            "VUL_StrongRoof_LightWall",
            "VUL_StrongRoof_SalvageWall",
            "VUL_LightRoof_StrongWall",
            "VUL_LightRoof_LightWall",
            "VUL_SalvagedRoof_StrongWall",
            "VUL_SalvagedRoof_LightWall",
            "VUL_SalvagedRoof_SalvageWall",
            "VUL_vulnerable_groups",
            "VUL_pantawid_pamilya_beneficiary",
        ]

        # split data into train and test sets

        SEED2 = 314159265
        SEED = 31

        test_size = 0.1

        # Full dataset for feature selection

        combined_input_data_ = combined_input_data[
            combined_input_data["DAM_perc_dmg"].notnull()
        ]

        X = combined_input_data_[selected_features_xgb_regr]
        y = combined_input_data_["DAM_perc_dmg"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=SEED
        )
        # from sklearn.metrics import mean_absolute_error
        reg = xgb.XGBRegressor(
            base_score=0.5,
            booster="gbtree",
            subsample=0.8,
            eta=0.05,
            max_depth=8,
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=1,
            early_stopping_rounds=10,
            eval_metric=mean_absolute_error,
            gamma=1,
            objective="reg:squarederror",
            gpu_id=-1,
            grow_policy="depthwise",
            learning_rate=0.025,
            min_child_weight=1,
            n_estimators=100,
            random_state=42,
            tree_method="hist",
        )

        eval_set = [(X_train, y_train), (X_test, y_test)]

        reg.fit(X, y, eval_set=eval_set)
        df_total['HAZ_v_max']=self.ECMWF_CORRECTION_FACTOR*df_total['HAZ_v_max']
        
        X_all = df_total[selected_features_xgb_regr]
        y_pred = reg.predict(X_all)
        y_pred[y_pred<0]=0
        df_total['Damage_predicted']=y_pred
        y_pred[y_pred<10]=0
        y_pred[y_pred>0]=1
        df_total['Trigger']=y_pred 
        
        df_total.loc[df_total['HAZ_dis_track_min'] > self.Wind_damage_radius, 'Damage_predicted'] = 0 # 
        
        df_total["dist_50"] = df_total["HAZ_dis_track_min"].apply(lambda x: 1 if x < 50 else 0)
        probability_impact=df_total.groupby('Mun_Code').agg(
                                    dist50k=('dist_50', sum),
                                    Num_ens=('dist_50', 'count')).reset_index()
        probability_impact["prob_within_50km"] = probability_impact.apply(lambda x: x.dist50k/x.Num_ens,axis=1)


        df_total = pd.merge(df_total,probability_impact.filter(['prob_within_50km','Mun_Code']),how="left",
            left_on="Mun_Code",
            right_on="Mun_Code")
        final_df=df_total.filter(["Damage_predicted", "Trigger","Mun_Code", "storm_id","ens_id", "HAZ_dis_track_min","HAZ_v_max","is_ensamble","prob_within_50km"])
        
        
        
        return final_df
        

                
        
    def set_zeros(self, x):
        x_max = 25
        y_max = 50
        v_max = x[0]
        rainfall_max = x[1]
        damage = x[2]
        Growth_factor=x[3]
        if pd.notnull(damage) and v_max > x_max:
            value = damage/Growth_factor
        #elif v_max > x_max or rainfall_max > y_max:
        #    value = damage
        elif v_max < x_max:#np.sqrt((1 - (rainfall_max**2 / y_max**2)) * x_max**2):
            value = 0
        # elif ((v_max < x_max)  and  (rainfall_max_6h < y_max) ):
        # elif (v_max < x_max ):
        # value = 0
        else:
            value = np.nan
        return value

    def pre_disaster_data(self):
        pre_disaster_inds = pd.read_csv(PRE_DISASTER_INDICATORS)
        pre_disaster_inds["vulnerable_groups"] = (
            pre_disaster_inds["vulnerable_groups"]
            .div(0.01 * pre_disaster_inds["Total Pop"], axis=0)
            .values
        )
        pre_disaster_inds["pantawid_pamilya_beneficiary"] = (
            pre_disaster_inds["pantawid_total_pop"]
            .div(0.01 * pre_disaster_inds["Total Pop"], axis=0)
            .values
        )
        pre_disaster_inds.rename(
            columns={
                "landslide_per": "GEN_landslide_per",
                "stormsurge_per": "GEN_stormsurge_per",
                "Bu_p_inSSA": "GEN_Bu_p_inSSA",
                "Bu_p_LS": "GEN_Bu_p_LS",
                "Red_per_LSbldg": "GEN_Red_per_LSbldg",
                "Or_per_LSblg": "GEN_Or_per_LSblg",
                "Yel_per_LSSAb": "GEN_Yel_per_LSSAb",
                "RED_per_SSAbldg": "GEN_RED_per_SSAbldg",
                "OR_per_SSAbldg": "GEN_OR_per_SSAbldg",
                "Yellow_per_LSbl": "GEN_Yellow_per_LSbl",
                "mean_slope": "TOP_mean_slope",
                "mean_elevation_m": "TOP_mean_elevation_m",
                "ruggedness_stdev": "TOP_ruggedness_stdev",
                "mean_ruggedness": "TOP_mean_ruggedness",
                "slope_stdev": "TOP_slope_stdev",
                "poverty_perc": "VUL_poverty_perc",
                "with_coast": "GEN_with_coast",
                "coast_length": "GEN_coast_length",
                "Housing Units": "VUL_Housing_Units",
                "Strong Roof/Strong Wall": "VUL_StrongRoof_StrongWall",
                "Strong Roof/Light Wall": "VUL_StrongRoof_LightWall",
                "Strong Roof/Salvage Wall": "VUL_StrongRoof_SalvageWall",
                "Light Roof/Strong Wall": "VUL_LightRoof_StrongWall",
                "Light Roof/Light Wall": "VUL_LightRoof_LightWall",
                "Light Roof/Salvage Wall": "VUL_LightRoof_SalvageWall",
                "Salvaged Roof/Strong Wall": "VUL_SalvagedRoof_StrongWall",
                "Salvaged Roof/Light Wall": "VUL_SalvagedRoof_LightWall",
                "Salvaged Roof/Salvage Wall": "VUL_SalvagedRoof_SalvageWall",
                "vulnerable_groups": "VUL_vulnerable_groups",
                "pantawid_pamilya_beneficiary": "VUL_pantawid_pamilya_beneficiary",
            },
            inplace=True,
        )
        return pre_disaster_inds
    
    
    def Number_affected(self,buildings):
        '''
        calclate the number of affected population
        '''
        import math
        import numpy as np
        if math.isnan(buildings):
            return np.nan
        else:            
            Number_affected_pop=int(np.exp(6.80943612231606) * buildings ** 0.46982114400549513)
            return Number_affected_pop
    
    def windfieldData(self, typhoons):
        Made_land_fall=0
        tr_HRS = [
            tr
            for tr in self.fcast_data
            if (tr.is_ensemble == "False" and tr.name == typhoons)
        ]
        fcast_data_typ = [ tr  for tr in self.fcast_data  if (tr.name in [typhoons]) ]

        track1 = TCTracks()
        track1.data = tr_HRS
        try:
            logger.info("High res member: creating intensity plot")
            fig, ax = plt.subplots(
                figsize=(12, 8), subplot_kw=dict(projection=ccrs.PlateCarree())
            )
            track1.plot(axis=ax)
            output_filename = os.path.join(Output_folder, f"track_{typhoons}")
            fig.savefig(output_filename)
        except:
            logger.info(f"incomplete track data ")

        if tr_HRS != []:
            HRS_SPEED = (
                tr_HRS[0].max_sustained_wind.values / 0.88
            ).tolist()  ############# 0.84 is conversion factor for ECMWF 10MIN TO 1MIN AVERAGE
            dfff = tr_HRS[0].to_dataframe()
            dfff[["VMAX", "LAT", "LON"]] = dfff[["max_sustained_wind", "lat", "lon"]]
            dfff["YYYYMMDDHH"] = dfff.index.values
            dfff["YYYYMMDDHH"] = dfff["YYYYMMDDHH"].apply(
                lambda x: x.strftime("%Y%m%d%H%M")
            )
            dfff["STORMNAME"] = typhoons
            hrs_track_data = dfff[["YYYYMMDDHH", "VMAX", "LAT", "LON", "STORMNAME"]]
            
            hrs_track_df=hrs_track_data.copy()#.query('5 < LAT < 20 and 115 < LON < 133')
            
            hrs_track_df.reset_index(inplace=True)
            coord_lat = gpd.GeoDataFrame(hrs_track_df.VMAX, geometry=gpd.points_from_xy(hrs_track_df.LON,hrs_track_df.LAT))
            coord_lat.set_crs(epsg=4326, inplace=True)
            
            
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            PHL = world[world['name'] == "Philippines"].dissolve(by='name')
            bounds1=(115.0,5.0,133.0,20.0)
            coast =coordinates.get_coastlines(bounds1, 10).geometry
            
            coastline = gpd.clip(coast.to_crs('EPSG:3124'), PHL.to_crs('EPSG:3124'))#.to_crs('EPSG:3124')
            
            points_df = coord_lat.to_crs('EPSG:3124') 
            points_df['min_dist_to_coast'] = points_df.geometry.apply(self.min_distance, args=(coastline,))

            MIN_DIST_TO_COAST = min(points_df['min_dist_to_coast'])

            #DIST_TO_COAST_M=coordinates.dist_to_coast(coord_lat, lon=None, signed=False)
             
            self.hrs_track_data[typhoons] = hrs_track_data
            
            #dfff[["YYYYMMDDHH", "VMAX", "LAT", "LON", "STORMNAME"]].to_csv(
            #    os.path.join(self.Input_folder, f"{typhoons}_ecmwf_hrs_track.csv"),
            #    index=False)
            
            typhoon_tracks=dfff[["YYYYMMDDHH", "VMAX", "LAT", "LON", "STORMNAME"]]
            typhoon_tracks["timestampOfTrackpoint"] = pd.to_datetime(
                typhoon_tracks["YYYYMMDDHH"], format="%Y%m%d%H%M"
            ).dt.strftime("%m-%d-%Y %H:%M:%S")
            
            typhoon_tracks["HH"] = pd.to_datetime(
                typhoon_tracks["YYYYMMDDHH"], format="%Y%m%d%H%M"
            ).dt.strftime("%H:%M")
            time_steps=['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00']
            typhoon_tracks=typhoon_tracks.query('HH in @time_steps')
            typhoon_tracks.to_csv(os.path.join(self.Input_folder, f"{typhoons}_ecmwf_hrs_track.csv"),index=False) 
            
            typhoon_tracks.rename(columns={"LON": "lon", "LAT": "lat"}, inplace=True)
            
            wind_tracks_hrs = typhoon_tracks[["lon", "lat", "timestampOfTrackpoint","HH","VMAX"]]
            wind_tracks_hrs.dropna(inplace=True)
            
            wind_tracks_hrs = wind_tracks_hrs.round(2)
            

                        
        

            # Adjust track time step
            data_forced = [
                tr.where(tr.time <= max(tr_HRS[0].time.values), drop=True)
                for tr in fcast_data_typ
            ]
            
            '''
            landfall_location_ = {}

            check_flag = 0


            for row, data in hrs_track_data.iterrows():
                p1 = Point(data["LAT"], data["LON"])
                self.admin["landfall"] = self.admin.contains(p1)
                landfall_ = self.admin.query('landfall=="True"')
                landfall_time = data["YYYYMMDDHH"]
                if not landfall_.empty and check_flag == 0:
                    landfall_location_[landfall_time] = landfall_
                    check_flag = check_flag + 1

            self.landfall_location[typhoons] = landfall_location_
            '''
            
            
            logger.info("CHECK LAND FALL TIME FINAL STEP")
            
            hrs_track_df = hrs_track_data.copy()    
            logger.info('calculating landfall time')
                     
            hrs_track_df["time"] = pd.to_datetime(hrs_track_df["YYYYMMDDHH"], format="%Y%m%d%H%M").dt.strftime("%Y-%m-%d %H:%M:%S")
        
            forecast_time = str(hrs_track_df['time'][0])
            
            forecast_time = datetime.strptime(forecast_time, "%Y-%m-%d %H:%M:%S")
                        ### filter areas in Philippiness area of responsibility
                        
            land=[]
            admin1=self.admin
            admin1=admin1.buffer(0)
            
            hrs_track_df_=hrs_track_df.copy()
            
            hrs_track_df_['geometry'] = [Point(xy) for xy in zip(hrs_track_df_.LON, hrs_track_df_.LAT)] 

            for row, data in hrs_track_df_.iterrows():
                p1 = data["geometry"]
                if any(list(set(admin1.contains(p1)))) == True:
                    land.append("land")
                else:
                    land.append("water") 
                    
            hrs_track_df_['location']=land
            
            is_onland='land'
            
            
            hrs_track_df_2=hrs_track_df_.query('location==@is_onland')
            
            point_inland_list=hrs_track_df_2['time'].values
            
            logger.info(f'land points {point_inland_list}')
          
            if len(point_inland_list)>0:
                landfalltime=min(point_inland_list)
                landfalltime_time_obj = datetime.strptime(str(landfalltime), "%Y-%m-%d %H:%M:%S")
                landfall_dellta = landfalltime_time_obj - forecast_time  # .strftime("%Y%m%d")
                #landfall_dellta = landfalltime_time_obj - datetime.utcnow()-9  # .strftime("%Y%m%d") 9hr for the latency 
                logger.info(f"CHECK LAND FALL TIME FINAL STEP {landfall_dellta}")
                seconds = landfall_dellta.total_seconds()
                hours = int(seconds // 3600)- self.ECMWF_LATENCY_LEADTIME_CORRECTION
                if hours < 0:
                    hours=0
                    Made_land_fall=1
                landfall_time_hr = str(hours) + "-hour"
            else:
                landfall_time_hr = "72-hour"     
                
                
            
            logger.info('calculating distance to land')
 
            coord_lat = gpd.GeoDataFrame(
                hrs_track_df.time,
                geometry=gpd.points_from_xy(hrs_track_df.LON, hrs_track_df.LAT), )
            
            coord_lat.set_crs(epsg=4326, inplace=True)
            
            DIST_TO_COAST_M = coordinates.dist_to_coast(coord_lat, lon=None, signed=False)

            hrs_track_df["distance"] = DIST_TO_COAST_M
            
            #hrs_track_df.reset_index(inplace=True)  
            logger.info(f'finished calculating distance to land{MIN_DIST_TO_COAST}')     
            if not wind_tracks_hrs.empty:    
                wind_tracks_hrs['KPH']=wind_tracks_hrs.apply(lambda x: self.ECMWF_CORRECTION_FACTOR*3.6*x.VMAX,axis=1)
                bins = [0,62,88, 117, 185, np.inf]
                catagories = ['TD', 'TS', 'STS', 'TY', 'STY']

                wind_tracks_hrs['catagories'] = pd.cut(wind_tracks_hrs['KPH'], bins, labels=catagories)
                exposure_place_codes=[]
                
                for ix, row in wind_tracks_hrs.iterrows():
                    if row["HH"] in ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00']:                        
                        exposure_entry = {
                            "lat": row["lat"],
                            "lon": row["lon"],
                            "windspeed":int(row["KPH"]),
                            "category":row["catagories"],
                            "timestampOfTrackpoint": row["timestampOfTrackpoint"],
                        }
                        exposure_place_codes.append(exposure_entry)

                json_file_path = self.Output_folder + typhoons + "_tracks" + ".json"
                
                track_records = {
                    "countryCodeISO3": "PHL",
                    "leadTime": landfall_time_hr,
                    "eventName": typhoons,
                    "trackpointDetails": exposure_place_codes,
                }

                with open(json_file_path, "w") as fp:
                    json.dump(track_records, fp)
                       

            # calculate wind field for each ensamble members
            if data_forced:# and MIN_DIST_TO_COAST <200000:#in meters            
                tracks = TCTracks()
                tracks.data =data_forced # 
                #tracks.equal_timestep(0.5)
                TYphoon = TropCyclone()
                TYphoon.set_from_tracks(tracks, self.cent, store_windfields=True,metric="geosphere")            
                windfield=TYphoon.windfields
                threshold =self.WIND_SPEED_THRESHOLD# 20
                list_intensity = []
                distan_track = []
                logger.info(f'finished calculating distance to land{MIN_DIST_TO_COAST}')  
                for i in range(len(data_forced)):
                    logger.info(f'finished calculating wind data for {i}')  
                    nsteps = windfield[i].shape[0]
                    tr=tracks.data[i]
                    centroid_id = np.tile(self.centroid_idx, nsteps)
                    intensity_3d = windfield[i].toarray().reshape(nsteps,  self.ncents, 2)
                    intensity = np.linalg.norm(intensity_3d, axis=-1).ravel()
                    timesteps = np.repeat(tracks.data[i].time.values, self.ncents)
                    timesteps = timesteps.reshape((nsteps,  self.ncents)).ravel()
                    inten_tr = pd.DataFrame({
                            'centroid_id': centroid_id,
                            'value': intensity,
                            'timestamp': timesteps,})
                    inten_tr = inten_tr[inten_tr.value > self.WIND_SPEED_THRESHOLD]
                    inten_tr['storm_id'] = tr.sid
                    inten_tr['name'] = tr.name
                    inten_tr = (pd.merge(inten_tr,  self.df_admin, how='outer', on='centroid_id')
                                .dropna()
                                .groupby(['adm3_pcode'], as_index=False)
                                .agg({"value": ['count', 'max']}))
                    inten_tr.columns = [x for x in ['adm3_pcode', 'value_count', 'v_max']]
                    inten_tr['storm_id'] = tr.sid
                    inten_tr['name'] = tr.name
                    inten_tr['forecast_time']=tr.forecast_time
                    #inten_tr['lead_time']=lead_time1
                    inten_tr["ens_id"] = tr.sid + "_" + str(tr.ensemble_number)
                    inten_tr['is_ensamble'] = tr.is_ensemble
                    list_intensity.append(inten_tr)
                    distan_track1=[]
                    for index, row in self.dfGrids.iterrows():
                        dist=np.min(np.sqrt(np.square(tr.lat.values-row['lat'])+np.square(tr.lon.values-row['lon'])))
                        distan_track1.append(dist*111)
                    dist_tr = pd.DataFrame({'centroid_id': self.centroid_idx,'value': distan_track1})
                    dist_tr = (pd.merge(dist_tr, self.df_admin, how='outer', on='centroid_id')
                                .dropna()
                                .groupby(['adm3_pcode'], as_index=False)
                                .agg({'value': 'min'}))
                    dist_tr.columns = [x for x in ['adm3_pcode', 'dis_track_min']]  # join_left_df_.columns.ravel()]
                    dist_tr['storm_id'] = tr.sid	
                    distan_track.append(dist_tr)
                    
                df_intensity_ = pd.concat(list_intensity)
                distan_track1 = pd.concat(distan_track)
                typhhon_df =  pd.merge(df_intensity_, distan_track1,  how='left', on=['adm3_pcode','storm_id'])
                
                if len(typhhon_df.index > 1):
                    typhhon_df.rename(
                        columns={"v_max": "HAZ_v_max", "dis_track_min": "HAZ_dis_track_min"},
                        inplace=True,
                    )
                    typhhon_wind_data = typhhon_df
                    typhhon_df['lead_time_hr']=landfall_time_hr
                    #Activetyphoon.append(typhoons)
                    
                    typhhon_df.to_csv(
                        os.path.join(self.Input_folder, f"{typhoons}_windfield.csv"),
                        index=False)
                    
                            # windspeed
                    layer='windspeed'
                    df_wind=typhhon_df
                    df_wind.fillna(0, inplace=True)
                    df_wind['windspeed']=df_wind['HAZ_v_max']
                    
                    df_wind.astype({"windspeed": "int32"})

                    exposure_data = {"countryCodeISO3": "PHL"}
                    exposure_place_codes = []
                    #### change the data frame here to include impact
                    for ix, row in df_wind.iterrows():
                        exposure_entry = {"placeCode": row["adm3_pcode"], "amount": int(row[layer])}
                        exposure_place_codes.append(exposure_entry)

                    exposure_data["exposurePlaceCodes"] = exposure_place_codes
                    exposure_data["adminLevel"] = self.admin_level
                    exposure_data["leadTime"] = landfall_time_hr
                    exposure_data["dynamicIndicator"] = layer
                    exposure_data["disasterType"] = "typhoon"
                    exposure_data["eventName"] = typhoons

                    json_file_path = self.Output_folder + typhoons + f"_{layer}" + ".json"

                    with open(json_file_path, "w") as fp:
                        json.dump(exposure_data, fp)       
            
                else:
                    Made_land_fall=2
                    
                    df_total_upload=self.pcode.copy()  #data frame with pcodes  
                    df_total_upload['alert_threshold']=0
                    df_total_upload['affected_population']=0   
                    df_total_upload['windspeed']=None 
                    df_total_upload['houses_affected']=None
                    df_total_upload['show_admin_area']=1
                    df_total_upload['prob_within_50km']=None
                    df_total_upload['rainfall']=None
                     

                                  
                    for layer in ["windspeed","rainfall", "prob_within_50km","houses_affected","affected_population","show_admin_area","alert_threshold"]:
                        exposure_entry=[]
                        # prepare layer
                        logger.info(f"preparing data for {layer}")
                        #exposure_data = {'countryCodeISO3': countrycode}
                        exposure_data = {"countryCodeISO3": "PHL"}
                        exposure_place_codes = []
                        #### change the data frame here to include impact
                        for ix, row in df_total_upload.iterrows():
                            exposure_entry = {"placeCode": row["adm3_pcode"],
                                            "amount": row[layer]}
                            exposure_place_codes.append(exposure_entry)
                            
                        exposure_data["exposurePlaceCodes"] = exposure_place_codes
                        exposure_data["adminLevel"] = self.admin_level
                        exposure_data["leadTime"] = "72-hour" #landfall_time_hr
                        exposure_data["dynamicIndicator"] = layer
                        exposure_data["disasterType"] = "typhoon"
                        exposure_data["eventName"] = typhoons                     
                        json_file_path = self.Output_folder + typhoons + f"_{layer}" + ".json"
                        
                        with open(json_file_path, 'w') as fp:
                            json.dump(exposure_data, fp)
                    

            
        logger.info(f'finshed wind calculation') 
        return Made_land_fall
    
    
    def impact_model(self, typhoon_names,wind_data):
        selected_columns = [
            "adm3_pcode",   
            "storm_id",
            "ens_id",
            "name",
            #"HAZ_max_06h_rain",
            #"HAZ_rainfall_max_24h",
            "HAZ_v_max",
            "is_ensamble",
            "HAZ_dis_track_min",
        ]

        '''
        df_hazard = pd.merge(
            wind_data,
            self.rainfall_data,
            how="left",
            left_on="adm3_pcode",
            right_on="adm3_pcode",
        ).filter(selected_columns)
        '''
        landfall_time_hr = list(set(wind_data.lead_time_hr.values))[0]
        df_hazard =wind_data.filter(selected_columns)
        
        df_total = pd.merge(
            df_hazard,
            self.pre_disaster_inds,
            how="left",
            left_on="adm3_pcode",
            right_on="Mun_Code",
        )

        df_total = df_total[df_total["HAZ_v_max"].notnull()]
        
        
        ######## calculate probability for track within 50km
        #df_total["dist_50"] = df_total["HAZ_dis_track_min"].apply(lambda x: 1 if x < 50 else 0)
        
 
        ### Run ML model

        impact_data = self.model(df_total)
        
 

        #df_total["Damage_predicted"] = impact_data
        #df_total.loc[df_total['HAZ_dis_track_min'] < 150, 'Damage_predicted'] = 0
        
        df_impact_forecast = pd.merge(
            impact_data,#df_total[["Damage_predicted", "Mun_Code", "storm_id", "HAZ_dis_track_min","HAZ_v_max","is_ensamble"]],
            self.pre_disaster_inds[["VUL_Housing_Units", "Mun_Code"]],
            how="left",
            left_on="Mun_Code",
            right_on="Mun_Code",
        )
        
        #df_impact_forecast["Hu"] = 0.01 * df_impact_forecast["VUL_Housing_Units"]
        
        #impact_scenarios = ["Damage_predicted"]
        
        df_impact_forecast['Damage_predicted_num'] = df_impact_forecast.apply(lambda x: 0.01*x['Damage_predicted']*x['VUL_Housing_Units'], axis=1)
        df_impact_forecast["number_affected_pop__prediction"] = df_impact_forecast.apply(lambda x: self.Number_affected(x["Damage_predicted_num"]), axis=1).values
        
        df_impact_forecast=df_impact_forecast[~df_impact_forecast['Damage_predicted_num'].isna()]
        
        
        csv_file_test = self.Output_folder + "Average_Impact_full_" + typhoon_names + ".csv"        
        df_impact_forecast.to_csv(csv_file_test)
        '''
        df_impact_forecast.loc[:, impact_scenarios] = df_impact_forecast.loc[
            :, impact_scenarios
        ].multiply(df_impact_forecast.loc[:, "Hu"], axis="index")

  
        
        df_impact_forecast[impact_scenarios] = df_impact_forecast[
            impact_scenarios
        ].astype("int")
        '''        
        df_impact_forecast["Damage_predicted_num"] = df_impact_forecast["Damage_predicted_num"].astype("int")   
        df_impact_forecast["number_affected_pop__prediction"] = df_impact_forecast["Damage_predicted_num"].astype("int") 
        
        impact = df_impact_forecast.copy()
        
        check_ensamble=False
        impact_HRS = impact.query("is_ensamble==@check_ensamble").filter(["Damage_predicted","Mun_Code",
                                                                          "HAZ_dis_track_min","number_affected_pop__prediction",
                                                                          "Damage_predicted_num","HAZ_v_max",
                                                                          "prob_within_50km"]
                                                                         ) 
        
        
        impact =impact.groupby("Mun_Code").agg(
            {"Damage_predicted": "mean",
             "VUL_Housing_Units":"mean",
             "number_affected_pop__prediction":"mean",
             "HAZ_dis_track_min": "min",
             "HAZ_v_max":"max",
             "prob_within_50km":"mean"}
        )
        impact["number_affected_pop__prediction"] = impact["number_affected_pop__prediction"].astype("int") 
        
        impact['Damage_predicted_num'] = impact.apply(lambda x: 0.01*x['Damage_predicted']*x['VUL_Housing_Units'], axis=1)
        impact["Damage_predicted_num"] = impact["Damage_predicted_num"].astype("int")
        
          
        
        csv_file_test = self.Output_folder + "Average_Impact_" + typhoon_names + ".csv"        
        impact.to_csv(csv_file_test)
        
  
 
        impact_df1 =impact_HRS# pd.merge(impact2,probability_50km,how="left", left_on="Mun_Code", right_on="Mun_Code",    )
        
  
        
        impact_HRS.rename(
            columns={
                "Damage_predicted_num": "num_houses_affected",
                "Damage_predicted": "houses_affected",
                "HAZ_dis_track_min": "WEA_dist_track",
                "HAZ_v_max": "windspeed",
                "number_affected_pop__prediction":"affected_population",
            },
            inplace=True,
        )
        
        logger.info(f"{len(impact_HRS)}")
       
        impact_df = pd.merge(
            self.pcode,
            impact_HRS.filter(["Mun_Code", "prob_within_50km","houses_affected", "WEA_dist_track","windspeed","affected_population"]),
            how="left", left_on="adm3_pcode", right_on="Mun_Code"
        )
        impact_df["show_admin_area"] = impact_df["WEA_dist_track"].apply(lambda x: 1 if x < self.Show_Areas_on_IBF_radius else 0)
        
        impact_df = impact_df.fillna(0)
        impact_df = impact_df.drop_duplicates("Mun_Code")
        impact_df["alert_threshold"] = 0
        

        

        # ------------------------ calculate and plot probability National -----------------------------------
        
        dref_probabilities = {
            "100k": [100000, 0.5],
            "80k": [80000, 0.6],
            "70k": [70000, 0.7],
            "50k": [50000, 0.8],
            "30k": [30000, 0.95],
        }


        dref_probabilities_10 = {
                    "33": [33,'Moderate'],
                    "50": [50,'High'],
                    "70":[70,'Very High'],
                }


        cerf_probabilities = {
            "80k": [80000, 0.5],
            "50k": [50000, 0.6],
            "30k": [30000, 0.7],
            "10k": [10000, 0.8],
            "5k": [5000, 0.95],
        }
                
                
        START_probabilities = {
            'PH166700000':{
                "8k": [8000, 0.8],
                "10k": [10000, 0.8],
                "15k": [15000, 0.7],
                "20k": [20000, 0.5],
                "25k": [25000, 0.5],    
                
                },
            'PH021500000':{
                "16k": [16000, 0.8],
                "18k": [18000, 0.8],
                "20k": [20000, 0.7],
                "25k": [25000, 0.5],
                "28.5k": [28500, 0.5], 
        
                },
            'PH082600000':{
                "18k": [18000, 0.8],
                "20k": [20000, 0.8],
                "25k": [25000, 0.7],
                "30k": [30000, 0.6],
                "40k": [40000, 0.5],
                },
        }
        # Only select regions 5 and 8
        cerf_regions = ["PH05", "PH08","PH16"]
        
        ######## calculate probability for impact
       # probability_impact = df_impact_forecast.groupby("storm_id").agg(        {"Damage_predicted_num": "sum"}       )
        
        
        probability_impact=df_impact_forecast.groupby('ens_id').agg(
                NUmber_of_affected_municipality=('Mun_Code','count'),
                Total_buildings_ML_Model=('Damage_predicted_num', sum),
                Trigger_ML_Model=('Trigger', sum)).reset_index()
            
        ### DREF trigger based on 10% damage per manucipality  
        DREF_trigger_list_10={}
        
        probability_impact['Trigger']=probability_impact.apply(lambda x:1 if x.Trigger_ML_Model>2 else 0,axis=1)
        
        agg_impact_10 = probability_impact["Trigger"].values
        logger.info(f'calculate trigger threshold{len(agg_impact_10)}')
        if len(agg_impact_10)>0:
            trigger_stat_dref10 = 100*(sum(agg_impact_10) /len(agg_impact_10))
        else:
            trigger_stat_dref10 =0
        logger.info('finished  calculating trigger threshold')    
        EAP_TRIGGERED = "no"
        eap_status_bool=0
        Trigger_status=True
 
     

        for key, values in dref_probabilities_10.items(): 
            dref_trigger_status10 = {}
            thershold=values[1]
            if  (trigger_stat_dref10 > values[0]):
                trigger_stat_1=True
                EAP_TRIGGERED = "yes"
                eap_status_bool = 1
 
            else:
                trigger_stat_1=False
                EAP_TRIGGERED = "no"
    
            dref_trigger_status10['triggered_prob'] = thershold 
            dref_trigger_status10['EVENT'] = typhoon_names 
            dref_trigger_status10['trigger_stat'] = trigger_stat_1 
            DREF_trigger_list_10[key] = dref_trigger_status10  
            
        self.eap_status[typhoon_names] = EAP_TRIGGERED
        self.eap_status_bool[typhoon_names] = eap_status_bool  
        
        impact_df["alert_threshold"]=impact_df.apply(lambda x: eap_status_bool if (x.Mun_Code in self.Tphoon_EAP_Areas.Mun_Code.values) else 0, axis=1)
 
        json_file_path = (
            self.Output_folder + typhoon_names + "_dref_trigger_status_10_percent" + ".csv"
        )
        pd.DataFrame.from_dict(DREF_trigger_list_10, orient="index").query('trigger_stat==@Trigger_status').to_csv(
            json_file_path
        )
        
        #save to file 
        csv_file2 = self.Output_folder + "HRS_Impact_" + typhoon_names + ".csv"        
        impact_df.to_csv(csv_file2)
      
        
        #probability based on number of buildings 
        dref_trigger_status = {}
        
        
        agg_impact = probability_impact["Total_buildings_ML_Model"].values
    
        for key, values in dref_probabilities.items():
            
            trigger_stat = (
                sum(i > values[0] for i in agg_impact) / (len(agg_impact)) > values[1]
            )
            dref_trigger_status[key] = trigger_stat
            dref_trigger_status['threshold'] = sum(self.Population_Growth_factor*i > values[0] for i in agg_impact) / (len(agg_impact)) 
      

                

        json_file_path = (
            self.Output_folder + typhoon_names + "_dref_trigger_status_Num_Bldg" + ".csv"
        )
        pd.DataFrame.from_dict(dref_trigger_status, orient="index").to_csv(
            json_file_path
        )
        
        

        
        
        #START Trigger Status 
        start_trigger_status = {}
        
        provinces_names={'PH166700000':'SurigaoDeLnorte','PH021500000':'Cagayan','PH082600000':'EasternSamar'}        
        df_impact_forecast['Prov_Code']=df_impact_forecast.apply(lambda x:str(x.Mun_Code[:6])+'00000',axis=1)
        
        df_impact_forecast_start=df_impact_forecast.query('Prov_Code in @provinces_names.keys()')
        if not df_impact_forecast_start.empty:
       
            for provinces in provinces_names.keys():#['PH166700000','PH021500000','PH082600000']:
                triggers=START_probabilities[provinces]
                prov_name=provinces_names[provinces]
                df_trig=df_impact_forecast.query('Prov_Code==@provinces')
                if not df_trig.empty:  
                    probability_impact2=df_trig.groupby(['ens_id']).agg(
                        NUmber_of_affected_municipality=('Mun_Code','count'),
                        average_ML_Model=('Damage_predicted', 'mean'),
                        Total_affected_ML_Model=('number_affected_pop__prediction', sum),
                        Total_buildings_ML_Model=('Damage_predicted_num', sum)).sort_values(by='Total_buildings_ML_Model',ascending=False).reset_index()
                    ######## calculate probability for impact                
                    
                    agg_impact_prov = probability_impact2["Total_affected_ML_Model"].values        
                    thershold='NAN'               
                    

                    for key, values in triggers.items():
                    
                        start_trigger_status1 = {}
                        trigger_stat_prov = sum(i > values[0] for i in agg_impact_prov) /len(agg_impact_prov)     
                        #start_trigger_status1[key] = int(trigger_stat)
                        start_trigger_status1['thr_name']=key
                        start_trigger_status1['thr_value']=trigger_stat_prov

                        if values[1] < trigger_stat_prov:
                            trigger_stat_=True
                            thershold=key 
                        else:
                            trigger_stat_=False
                            thershold=key
                    #start_trigger_list.append({'event':event_name, 'province':prov_name,                                      'lead_time':lead_time2,                                      'status':trigger_stat_,                                      'threshold':thershold})
                    
                        #start_trigger_status['event'] = event_name
        

                        start_trigger_status1['PROVINCE'] = prov_name
                        start_trigger_status1['EVENT'] = typhoon_names       
                        start_trigger_status1['trigger_stat'] = trigger_stat_  
                        start_trigger_status[key] = start_trigger_status1
            
                    #start_trigger[prov_name] =start_trigger_status
                    
                #start_trigger_list.append(start_trigger_status)
            #start_trigger_list[event_name] = start_trigger_status
            
            json_file_path = (
                self.Output_folder + typhoon_names + "_start_trigger_status" + ".csv"
            )
            pd.DataFrame.from_dict(start_trigger_status, orient="index").to_csv(
                json_file_path
            )
        
        
                
        #CERF Trigger Status         
        cerf_trigger_status = {}
        
        df_impact_forecast["reg"] = df_impact_forecast["Mun_Code"].apply(
            lambda x: x[:4]
        )
        df_impact_forecast_cerf = df_impact_forecast.query("reg in @cerf_regions")
        
        if not df_impact_forecast_cerf.empty:            
            probability_impact = df_impact_forecast_cerf.groupby("ens_id").agg(
                {"Damage_predicted_num": "sum"}
            )
            
            probability_impact.reset_index(inplace=True)
            
            agg_impact = probability_impact["Damage_predicted_num"].values

            for key, values in cerf_probabilities.items():
                cerf_trigger_status[key] = (
                    sum(i > values[0] for i in agg_impact) / (len(agg_impact)) > values[1]
                )

            json_file_path = (
                self.Output_folder + typhoon_names + "_cerf_trigger_status" + ".csv"
            )
            pd.DataFrame.from_dict(cerf_trigger_status, orient="index").to_csv(
                json_file_path
            )
            

        #############################################################

        # adm3_pcode,storm_id,is_ensamble,value_count,v_max,name,dis_track_min
        typhoon_windAll = wind_data.copy()# self.typhoon_wind_data[typhoon_names]
        check_ensamble=False
        typhoon_windAll = typhoon_windAll.query('is_ensamble==@check_ensamble')

        typhoon_wind = typhoon_windAll[
            ["adm3_pcode", "HAZ_v_max", "HAZ_dis_track_min"]
        ].drop_duplicates("adm3_pcode").copy()
        typhoon_wind.rename(columns={"HAZ_v_max": "windspeed"}, inplace=True)
        logger.info(f"{len(typhoon_wind)}")

        # max_06h_rain,max_24h_rain,Mun_Code
        typhoon_rainfall = self.rainfall_data[["adm3_pcode", "HAZ_rainfall_max_24h"]].copy()
        typhoon_rainfall.rename(
            columns={"HAZ_rainfall_max_24h": "rainfall"},
            inplace=True,
        )
        logger.info(f"{len(typhoon_rainfall)}")

        # create dataframe for all manucipalities

        df_wind = pd.merge(
            self.pcode,
            typhoon_wind,
            how="left",
            left_on="adm3_pcode",
            right_on="adm3_pcode",
        )
  

        check_ensamble=False
        rain_df=self.rainfall_data.copy()
        
        df_hazard2=pd.merge(
            df_wind,
            typhoon_rainfall,
            how="left",
            left_on="adm3_pcode",
            right_on="adm3_pcode",
        ).filter(selected_columns)
        
        #df_hazard2=df_hazard2.query("is_ensamble==@check_ensamble")
        #df_hazard2.rename(columns={"HAZ_rainfall_max_24h": "windspeed"},inplace=True)
        
        df_hazard2=df_hazard2.filter(["adm3_pcode","rainfall", "windspeed"])
        
        # "","adm3_en","glat","adm3_pcode","adm2_pcode","adm1_pcode","glon","GEN_mun_code","probability_dist50","impact","WEA_dist_track"
        
        csv_file2 = self.Output_folder + "HRS_Impact_" + typhoon_names + ".csv"
 
        Model_output_data=pd.read_csv(csv_file2).filter(["adm3_pcode","prob_within_50km","houses_affected",
                                                         "alert_threshold","show_admin_area","affected_population"])
     
        df_total_upload = pd.merge(
            df_hazard2,
            Model_output_data,
            how="left",
            left_on="adm3_pcode",
            right_on="adm3_pcode")
        
        df_total_upload.fillna(0, inplace=True)
        logger.info(f"{len(df_total_upload)}")
        # "landfall_time": landfalltime_time_obj,"lead_time": landfall_time_hr,
        # "EAP_triggered": EAP_TRIGGERED}]

        typhoon_track = self.hrs_track_data[typhoon_names].copy()
        
        typhoon_track["timestampOfTrackpoint"] = pd.to_datetime(
            typhoon_track["YYYYMMDDHH"], format="%Y%m%d%H%M"
        ).dt.strftime("%m-%d-%Y %H:%M:%S")
        
        typhoon_track["HH"] = pd.to_datetime(
            typhoon_track["YYYYMMDDHH"], format="%Y%m%d%H%M"
        ).dt.strftime("%H:%M")
        
        typhoon_track.rename(columns={"LON": "lon", "LAT": "lat"}, inplace=True)
        wind_track = typhoon_track[["lon", "lat", "timestampOfTrackpoint","HH","VMAX"]]

        ##############################################################################
        # rainfall
        layer='rainfall'
        #typhoon_rainfall.astype({"rainfall": "int32"})

        exposure_data = {"countryCodeISO3": "PHL"}
        exposure_place_codes = []
        #### change the data frame here to include impact
        for ix, row in typhoon_rainfall.iterrows():
            exposure_entry = {"placeCode": row["adm3_pcode"], "amount": int(row[layer])}
            exposure_place_codes.append(exposure_entry)

        exposure_data["exposurePlaceCodes"] = exposure_place_codes
        exposure_data["adminLevel"] = self.admin_level
        exposure_data["leadTime"] = landfall_time_hr
        exposure_data["dynamicIndicator"] = 'rainfall'
        exposure_data["disasterType"] = "typhoon"
        exposure_data["eventName"] = typhoon_names

        json_file_path = self.Output_folder + typhoon_names + f"_{layer}" + ".json"

        with open(json_file_path, "w") as fp:
            json.dump(exposure_data, fp)        
        
        # windspeed
        layer='windspeed'
        df_wind.fillna(0, inplace=True)
        df_wind.astype({"windspeed": "int32"})

        exposure_data = {"countryCodeISO3": "PHL"}
        exposure_place_codes = []
        #### change the data frame here to include impact
        for ix, row in df_wind.iterrows():
            exposure_entry = {"placeCode": row["adm3_pcode"], "amount": int(row[layer])}
            exposure_place_codes.append(exposure_entry)

        exposure_data["exposurePlaceCodes"] = exposure_place_codes
        exposure_data["adminLevel"] = self.admin_level
        exposure_data["leadTime"] = landfall_time_hr
        exposure_data["dynamicIndicator"] = layer
        exposure_data["disasterType"] = "typhoon"
        exposure_data["eventName"] = typhoon_names

        json_file_path = self.Output_folder + typhoon_names + f"_{layer}" + ".json"

        with open(json_file_path, "w") as fp:
            json.dump(exposure_data, fp)          
        
        
        
        # track

        exposure_place_codes = []
        wind_track.dropna(inplace=True)
        wind_track = wind_track.round(2)
 
        wind_track['KPH']=wind_track.apply(lambda x: self.ECMWF_CORRECTION_FACTOR*3.6*x.VMAX,axis=1)

        

        
        
        
        '''
        TROPICAL DEPRESSION (TD) - a tropical cyclone with maximum sustained winds of up to 62 kilometers per hour (kph) or less than 34 nautical miles per hour (knots) .
        TROPICAL STORM (TS) - a tropical cyclone with maximum wind speed of 62 to 88 kph or 34 - 47 knots.
        SEVERE TROPICAL STORM (STS) , a tropical cyclone with maximum wind speed of 87 to 117 kph or 48 - 63 knots.
        TYPHOON (TY) - a tropical cyclone with maximum wind speed of 118 to 184 kph or 64 - 99 knots.
        SUPER TYPHOON (STY) - a tropical cyclone with maximum wind speed exceeding 185 kph or more than 100 knots.
        '''
        bins = [0,62,88, 117, 185, np.inf]
        catagories = ['TD', 'TS', 'STS', 'TY', 'STY']

        wind_track['catagories'] = pd.cut(wind_track['KPH'], bins, labels=catagories)
        
        for ix, row in wind_track.iterrows():
            if row["HH"] in ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00']:
                
                exposure_entry = {
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "windspeed":int(row["KPH"]),
                    "category":row["catagories"],
                    "timestampOfTrackpoint": row["timestampOfTrackpoint"],
                }
                exposure_place_codes.append(exposure_entry)

        json_file_path = self.Output_folder + typhoon_names + "_tracks" + ".json"

        track_records = {
            "countryCodeISO3": "PHL",
            "leadTime": landfall_time_hr,
            "eventName": typhoon_names,
            "trackpointDetails": exposure_place_codes,
        }

        with open(json_file_path, "w") as fp:
            json.dump(track_records, fp)

        # dynamic layers
        
        #df_total_upload = df_total_upload.astype({"prob_within_50km": "int32","houses_affected": "int32","alert_threshold": "int32","show_admin_area": "int32"})
      
        try:
            for layer in ["prob_within_50km","houses_affected","alert_threshold","show_admin_area","affected_population"]:
                # prepare layer
                logger.info(f"preparing data for {layer}")
                # exposure_data = {'countryCodeISO3': countrycode}
                exposure_data = {"countryCodeISO3": "PHL"}
                exposure_place_codes = []
                #### change the data frame here to include impact
                for ix, row in df_total_upload.iterrows():
                    if layer in ["houses_affected"]: #"prob_within_50km",
                        exposure_entry = {"placeCode": row["adm3_pcode"], "amount": round(0.01*row[layer],2)}
                    elif layer in ["prob_within_50km"]: #"",
                        exposure_entry = {"placeCode": row["adm3_pcode"], "amount": round(1*row[layer],2)}
                    else:
                        exposure_entry = {"placeCode": row["adm3_pcode"], "amount": int(row[layer])}
                    exposure_place_codes.append(exposure_entry)

                exposure_data["exposurePlaceCodes"] = exposure_place_codes
                exposure_data["adminLevel"] = self.admin_level
                exposure_data["leadTime"] = landfall_time_hr
                exposure_data["dynamicIndicator"] = layer
                exposure_data["disasterType"] = "typhoon"
                exposure_data["eventName"] = typhoon_names

                json_file_path = self.Output_folder + typhoon_names + f"_{layer}" + ".json"
                with open(json_file_path, "w") as fp:
                    json.dump(exposure_data, fp)
        except:
            logger.info(f"no data for {layer}")
            pass
        logger.info("finshed preparing data for all dynamic layers")
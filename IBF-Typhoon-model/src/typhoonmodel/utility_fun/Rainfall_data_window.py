from ftplib import FTP
import sys
import os
import re
import zipfile
from pybufrkit.renderer import FlatTextRenderer
from sys import platform
import urllib.request
import requests
from bs4 import BeautifulSoup
from os.path import relpath
import subprocess
from os import listdir
from os.path import isfile, join
import geopandas as gpd
import pandas as pd
from typhoonmodel.utility_fun.settings import *
import rasterio

import urllib.error
import requests
import logging
from pathlib import Path
import shutil
import xarray as xr
from rasterstats import zonal_stats



def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


    
def download_rainfall_nomads(no_data_value=29999):
    """
    download rainfall 
    
    """

    
    list_df=[]  #to store final rainfall dataframes 
    #ADMIN_PATH = 'data/gis_data/phl_admin3_simpl2.geojson'
    admin = gpd.read_file(ADMIN_PATH)
    #path_admin =os.path.join(path, 'data/gis-data/phl_admin3_simpl2.geojson')
    #admin = gpd.read_file(path_admin)
    rainfall_time_step=['06', '24']
 
    url='https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.%s/'% data_point #datetime.now().strftime("%Y%m%d")
    url2='https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.%s/'% Alternative_data_point #datetime.now().strftime("%Y%m%d")
    
    def listFD(url, ext=''):
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        return [url + node.get('href') for node in soup.find_all('a') if node.get('href').split('/')[-2] in ['00','06','12','18']]#.endswith(ext)]
    base_urls=[]
    for items in listFD(url, ext=''):
        if url_is_alive(items+'prcp_bc_gb2/'):
            base_urls.append(items)
    
    try:        
        base_url=base_urls[-1]
        base_url_hour=base_url+'prcp_bc_gb2/geprcp.t%sz.pgrb2a.0p50.bc_' % base_url.split('/')[-2]
        time_step_list=['06','12','18','24','30','36','42','48','54','60','66','72']
        time_step_list1=['24','30','36','42','48','54','60','66','72']
        rainfall_24=[base_url_hour+'24hf0%s'%t for t in time_step_list1]
        rainfall_06=[base_url_hour+'06hf0%s'%t for t in time_step_list]
        rainfall_24.extend(rainfall_06)
        for rain_file in rainfall_24:
            print(rain_file)
            output_file= os.path.join(relpath(rainfall_path,path),rain_file.split('/')[-1]+'.grib2')
            #output_file= os.path.join(rainfall_path,rain_file.split('/')[-1]+'.grib2')
            batch_ex="wget -O %s %s" %(output_file,rain_file)
            os.chdir(MAIN_DIRECTORY)
            print(batch_ex)
            p = subprocess.call(batch_ex ,cwd=MAIN_DIRECTORY)
    except:
        base_url=listFD(url2, ext='')[-1]
        base_url_hour=base_url+'prcp_bc_gb2/geprcp.t%sz.pgrb2a.0p50.bc_' % base_url.split('/')[-2]
        time_step_list=['06','12','18','24','30','36','42','48','54','60','66','72']
        time_step_list1=['24','30','36','42','48','54','60','66','72']
        rainfall_24=[base_url_hour+'24hf0%s'%t for t in time_step_list1]
        rainfall_06=[base_url_hour+'06hf0%s'%t for t in time_step_list]
        rainfall_24.extend(rainfall_06)
        for rain_file in rainfall_24:
            #output_file= os.path.join(rainfall_path,rain_file.split('/')[-1]+'.grib2')
            output_file= os.path.join(relpath(rainfall_path,MAIN_DIRECTORY),rain_file.split('/')[-1]+'.grib2')
            batch_ex="wget -O %s %s" %(output_file,rain_file)
            os.chdir(MAIN_DIRECTORY)
            print(batch_ex)
            p = subprocess.call(batch_ex ,cwd=MAIN_DIRECTORY)
        
    rain_files = [f for f in listdir(rainfall_path) if isfile(join(rainfall_path, f))]
    os.chdir(rainfall_path)
    pattern1='.pgrb2a.0p50.bc_06h'
    pattern2='.pgrb2a.0p50.bc_24h'
    for files in rain_files:
        if pattern2 in files:
            p = subprocess.call('wgrib2 %s -append -netcdf rainfall_24.nc'%files )
            os.remove(files)
        if pattern1 in files:
            p = subprocess.call('wgrib2 %s -append -netcdf rainfall_06.nc'%files )
            os.remove(files)
    #zonal stats to calculate rainfall per manucipality 
    rainfiles = [f for f in os.listdir(rainfall_path) if f.endswith('.nc') ]
    col_names=[ f.split('_')[1][0:2] for f in rainfiles]
    for layer in rainfiles:
        rain_6h=rasterio.open(os.path.join(rainfall_path,layer))         
        band_indexes = rain_6h.indexes
        transform = rain_6h.transform
        all_band_summaries = []
        for b in band_indexes:
            array = rain_6h.read(b)
            band_summary = zonal_stats(
                admin,
                array,
                prefix=f"band{b}_",
                stats="mean",
                nodata=no_data_value,
                all_touched=True,
                affine=transform,
            )
            all_band_summaries.append(band_summary)
        # Flip dimensions
        shape_summaries = list(zip(*all_band_summaries))
        # each list entry now reflects a municipalities, and consists of a dictionary with the rainfall in mm / 6h for each time frame
        final = [{k: v for d in s for k, v in d.items()} for s in shape_summaries]
        # Obtain list with maximum 6h rainfall
        maximum_6h = [max(x.values()) for x in final]
        list_df.append(pd.DataFrame(maximum_6h))
    df_rain = pd.concat(list_df,axis=1, ignore_index=True) 
    df_rain.columns = ["max_"+time_itr+"h_rain" for time_itr in col_names]
    df_rain['Mun_Code']=list(admin['adm3_pcode'].values)
    df_rain.to_csv(os.path.join(rainfall_path, "rain_data.csv"), index=False)
    os.chdir(MAIN_DIRECTORY)

def download_rainfall_DWD():
    """
    download rainfall 
    
    """

    time_step_list=['006','012','018','024','030','036','042','048','054','060','066','072']
    if datetime.now().hour >17:
      base_url='http://opendata.dwd.de/weather/wmc/icon-eps/data/grib/'+'fc_%s12icgle_'%datetime.now().strftime("%Y%m%d")
    else:
      base_url='http://opendata.dwd.de/weather/wmc/icon-eps/data/grib/'+'fc_%s00icgle_'%datetime.now().strftime("%Y%m%d")
  
    rainfall_24=[base_url+'%s.grib'%t for t in time_step_list]
    print(rainfall_24)
    try:
      for rain_file in rainfall_24:
        local_filename= os.path.join(rainfall_path,rain_file.split('/')[-1])
        print(rain_file)
        urllib.request.urlretrieve (rain_file,local_filename)
    except:
      print('download failed')
      pass
      
def zonal_stat_rain(filepath,admin):
    #zonal stats to calculate rainfall per manucipality 
    rain_6h=rasterio.open(filepath) 
    logger.info("Open rainfll necdf files to perfomrm zonal statistics")
    band_indexes = rain_6h.indexes
    transform = rain_6h.transform
    all_band_summaries = []
    for b in band_indexes:
        array = rain_6h.read(b)
        band_summary = zonal_stats(
            admin,
            array,
            prefix=f"band{b}_",
            stats="mean",
            nodata=no_data_value,
            all_touched=True,
            affine=transform,
        )
        all_band_summaries.append(band_summary)
    # Flip dimensions
    shape_summaries = list(zip(*all_band_summaries))
    # each list entry now reflects a municipalities, and consists of a dictionary with the rainfall in mm / 6h for each time frame
    final = [{k: v for d in s for k, v in d.items()} for s in shape_summaries]
    # Obtain list with maximum 6h rainfall
    maximum_6h = [max(x.values()) for x in final]
    return pd.DataFrame(maximum_6h)
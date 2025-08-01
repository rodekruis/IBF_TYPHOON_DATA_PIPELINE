import os
import urllib.request
import urllib.error
import requests
import logging
from pathlib import Path
import shutil

from bs4 import BeautifulSoup
import xarray as xr
import rasterio
from rasterstats import zonal_stats
import geopandas as gpd
import pandas as pd
from typhoonmodel.utility_fun.settings import *
from affine import Affine
import rioxarray as rio
import numpy as np
import netCDF4

logger = logging.getLogger(__name__)


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
    except urllib.error.HTTPError:
        return False


def download_rainfall_nomads( no_data_value=29999):
    """
    download rainfall 
    """

    list_df=[]  #to store final rainfall dataframes 
    #ADMIN_PATH = 'data/gis_data/phl_admin3_simpl2.geojson'
    admin = gpd.read_file(ADMIN_PATH)
    RAINFALL_TIME_STEP=['06', '24']

    url_base = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/naefs/prod/gefs.'

    #url1 = f"{url_base}{data_point}/"  # Use the timestamp of the input folder for the query
    #url2 = f"{url_base}{Alternative_data_point}/"  # Yesterday's date


    try:
        logger.info("Trying to get rainfall from today's date")
        url1=[f'{url_base}{data_point}/{hh}/' for hh in ['00','06','12','18']]
        get_grib_files(url1, MAIN_DIRECTORY, rainfall_path)
    except IndexError:
        # If list index out of range then it means that there are no files available,
        # use tomorrow's date instead
        logger.warning(f"No rainfall files available today, using yesterday's date instead")
        url2=[f'{url_base}{Alternative_data_point}/{hh}/' for hh in ['00','06','12','18']]
        get_grib_files(url2, MAIN_DIRECTORY, rainfall_path)

    rain_dict={}

    for hour in RAINFALL_TIME_STEP:
        pattern = f'.pgrb2a.0p50.bc_{hour}h'
        output_filename = f'rainfall_{hour}.nc'
        filename_list = Path(rainfall_path).glob(f'*{pattern}*')
        with xr.open_mfdataset(filename_list, engine='cfgrib',
                                combine="nested", concat_dim=["time"],
                                backend_kwargs={"indexpath": "",
                                                'filter_by_keys': {'totalNumber': 30}}
                                ) as ds:
            filepath = os.path.join(rainfall_path, output_filename)
        
            ds = ds.median(dim='number') #store only the median of the ensemble members
            ds.to_netcdf(filepath)
        #zonal stats to calculate rainfall per manucipality 
        #list_df.append(zonal_stat_rain(filepath,admin))  
        ds = xr.open_dataset(filepath)
        rain_var = ds[list(ds.data_vars)[0]]  # or use ds['rainfall'] if known
        rain_data = rain_var.load()  # Ensure data is loaded into memory
        rain_data.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
        rain_data.rio.write_crs("EPSG:4326", inplace=True)
        # Get coordinate values
        lat = rain_data.latitude.values
        lon = rain_data.longitude.values

        if lat[0] < lat[-1]:
            rain_data = rain_data.sortby("latitude", ascending=False)
            lat = rain_data.latitude.values

        # Create affine transform (from top-left corner)
        res_lat = abs(lat[1] - lat[0])
        res_lon = abs(lon[1] - lon[0])
        transform = Affine.translation(lon[0] - res_lon / 2, lat[0] - res_lat / 2) * Affine.scale(res_lon, -res_lat)


        # Loop over each time step and compute zonal mean
        list_df = []
        for i in range(rain_data.sizes["time"]):
            array = rain_data.isel(time=i).values  # shape: (lat, lon)

            band_summary = zonal_stats(
                admin,                  # GeoDataFrame of admin areas
                array,
                stats="mean",
                nodata=np.nan,
                all_touched=True,
                affine=transform
            )
                # Extract means
            values = [stat['mean'] if stat['mean'] is not None else 0 for stat in band_summary]
            
            list_df.append(pd.DataFrame(values, columns=[f"timestep_{i}"]))
            df_combined = pd.concat(list_df, axis=1)
            df_max = df_combined.max(axis=1)
            rain_dict[f"max_{hour}h_rain"] = df_max.values

    df_rain = pd.DataFrame(rain_dict)
    df_rain['Mun_Code'] = list(admin['adm3_pcode'].values)
    df_rain.to_csv(os.path.join(rainfall_path, "rain_data.csv"), index=False)

''' 

    for hour in RAINFALL_TIME_STEP:
        pattern = f'.pgrb2a.0p50.bc_{hour}h'
        output_filename = f'rainfall_{hour}.nc'
        filename_list = Path(rainfall_path).glob(f'*{pattern}*')
        with xr.open_mfdataset(filename_list, engine='cfgrib',
                               combine="nested", concat_dim=["time"],
                               backend_kwargs={"indexpath": "",
                                               'filter_by_keys': {'totalNumber': 30}}
                               ) as ds:
            filepath = os.path.join(rainfall_path, output_filename)
            logger.info(f'Writing to file {filepath}')
            ds = ds.median(dim='number') #store only the median of the ensemble members
            ds.to_netcdf(filepath)
        #zonal stats to calculate rainfall per manucipality 
        #list_df.append(zonal_stat_rain(filepath,admin))  
        rain_6h=rioxarray.open_rasterio(filepath) #rasterio.open(filepath) 
        rain_6h.rio.write_crs("EPSG:4326", inplace=True)  # Ensure CRS is set

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
    df_rain.columns = ["max_"+time_itr+"h_rain" for time_itr in RAINFALL_TIME_STEP]
    df_rain['Mun_Code']=list(admin['adm3_pcode'].values)
    logger.info("saved processed rainfall file to csv")
    df_rain.to_csv(os.path.join(rainfall_path, "rain_data.csv"), index=False)
'''



    
    #df_rain = pd.concat(list_df,axis=1, ignore_index=True) 
    #df_rain.columns = ["max_"+time_itr+"h_rain" for time_itr in RAINFALL_TIME_STEP]
    #df_rain['Mun_Code']=list(admin['adm3_pcode'].values)
    #df_rain.to_csv(os.path.join(Input_folder, "rainfall/rain_data.csv"), index=False)
    #logger.info("saved processed rainfall file to csv")


def get_grib_files(url, path, rainfall_path, use_cache=True):
    base_urls = []
    for items in url: #listFD(url):
        if url_is_alive(items+'prcp_bc_gb2/'):
            base_urls.append(items)
    base_url = base_urls[-1]
    base_url_hour = base_url+'prcp_bc_gb2/geprcp.t%sz.pgrb2a.0p50.bc_' % base_url.split('/')[-2]
    time_step_list1 = [ '24', '30', '36', '42', '48', '54', '60', '66', '72']
    time_step_list = ['06', '12', '18', '24', '30', '36', '42', '48', '54', '60', '66', '72']
    rainfall_24 = [base_url_hour+'24hf0%s' % t for t in time_step_list1]
    rainfall_06 = [base_url_hour+'06hf0%s' % t for t in time_step_list]
    for rain_file in rainfall_06 + rainfall_24:
        output_file = os.path.join(os.path.relpath(rainfall_path, path), rain_file.split('/')[-1]+'.grib2')
        if use_cache and os.path.isfile(output_file):
            logger.info(f'File {output_file} exists, skipping')
            continue
        try:
            with urllib.request.urlopen(rain_file) as response, open(output_file, 'wb') as out_file:
                logger.info(f'Downloading {rain_file} to {output_file}')
                shutil.copyfileobj(response, out_file)
        except urllib.error.HTTPError:
            logger.warning(f"Rain file {rain_file} doesn't exist, skipping")
            continue


def listFD(url):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + node.get('href')
            for node in soup.find_all('a')
            if node.get('href').split('/')[-2]
            in ['00', '06', '12', '18']]
            
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
    
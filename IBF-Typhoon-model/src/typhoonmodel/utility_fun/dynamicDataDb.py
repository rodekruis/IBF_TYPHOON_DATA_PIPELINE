import zipfile
import pandas as pd
import requests
import json
from typhoonmodel.utility_fun.settings import *
import os
import numpy as np
import logging
logger = logging.getLogger(__name__)

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


class DatabaseManager:

    """ Class to upload and process data in the database """

    def __init__(self, countryCodeISO3,admin_level):
        self.countryCodeISO3 = countryCodeISO3
        #self.leadTimeLabel = leadTimeLabel
        #self.triggerFolder = PIPELINE_OUTPUT + "triggers_rp_per_station/"
        #self.affectedFolder = PIPELINE_OUTPUT + "calculated_affected/"
        #self.EXPOSURE_DATA_SOURCES = SETTINGS[countryCodeISO3]['EXPOSURE_DATA_SOURCES']
        self.ADMIN_PASSWORD = SETTINGS_SECRET[countryCodeISO3]['ADMIN_PASSWORD']
        self.ADMIN_LOGIN=SETTINGS_SECRET[countryCodeISO3]["ADMIN_LOGIN"]
        self.API_SERVICE_URL = SETTINGS_SECRET[countryCodeISO3]['IBF_API_URL'] 
        self.admin_level = admin_level
        self.Output_folder = Output_folder
        
        self.mock=SETTINGS_SECRET[countryCodeISO3]["mock"]
        self.mock_nontrigger_typhoon_event=SETTINGS_SECRET[countryCodeISO3]["mock_nontrigger_typhoon_event"]
        self.mock_trigger_typhoon_event=SETTINGS_SECRET[countryCodeISO3]["mock_trigger_typhoon_event"]
        self.mock_trigger=SETTINGS_SECRET[countryCodeISO3]["if_mock_trigger"]
        
        self.uploadTime = uploadTime


 
        #ADMIN_LOGIN=SETTINGS_SECRET[countryCodeISO3]["ADMIN_LOGIN"]
        #ADMIN_PASSWORD=SETTINGS_SECRET[countryCodeISO3]["ADMIN_PASSWORD"]
        #IBF_API_URL=SETTINGS_SECRET[countryCodeISO3]["IBF_API_URL"]

    def upload(self):
        self.uploadTriggersPerLeadTime()
        self.uploadTriggerPerStation()
        self.uploadCalculatedAffected()
        self.uploadRasterFile()
    
    def processEvents(self):
        if SETTINGS_SECRET[self.countryCodeISO3]["notify_email"]:
            path = 'events/process' #default is noNotifications=false
        else:
            path = 'events/process?noNotifications=true'
        
        body = {
            'countryCodeISO3': self.countryCodeISO3,
            'disasterType': self.getDisasterType(),
            'date': self.uploadTime
            }
        self.apiPostRequest(path, body=body)
            
        logger.info('process events instructions sent')
    
    def getDisasterType(self):
        disasterType = "typhoon"
        return disasterType
        
    def uploadTyphoonData(self,json_path):  
        for indicator in ["windspeed","rainfall", "prob_within_50km","houses_affected","affected_population","show_admin_area","forecast_severity","forecast_trigger"]:
            json_file_path =json_path +f'_{indicator}' + '.json'
            try:
                with open(json_file_path) as json_file:
                    body = json.load(json_file)
                    body['date'] = self.uploadTime
                    #body['adminLevel'] = self.admin_level
                    self.apiPostRequest('admin-area-dynamic-data/exposure', body=body)                     
                logger.info(f'Uploaded data for indicator: {indicator} ')
            except requests.exceptions.ReadTimeout:
                logger.info(f'time out during Uploading data for indicator: {indicator} ')  
                pass    
    def uploadTyphoonDataAfterlandfall(self,json_path):  
        for indicator in ["prob_within_50km","houses_affected","affected_population","show_admin_area","forecast_severity","forecast_trigger"]:
            json_file_path =json_path +f'_{indicator}' + '.json'
            try:
                with open(json_file_path) as json_file:
                    body = json.load(json_file)
                    body['leadTime']= '0-hour'
                    body['date'] = self.uploadTime
                    self.apiPostRequest('admin-area-dynamic-data/exposure', body=body)                     
                logger.info(f'Uploaded data for indicator: {indicator} ')
            except requests.exceptions.ReadTimeout:
                logger.info(f'time out during Uploading data for indicator: {indicator} ')  
                pass                          
    def uploadTyphoonDataNoLandfall(self,json_path):  
        for indicator in ["windspeed","rainfall", "prob_within_50km","houses_affected","affected_population","show_admin_area","forecast_severity","forecast_trigger"]:
            json_file_path =json_path +f'_{indicator}' + '.json'
            try:
                with open(json_file_path) as json_file:
                    body = json.load(json_file)
                    #body['adminLevel'] = self.admin_level
                    body['date'] = self.uploadTime
                    self.apiPostRequest('admin-area-dynamic-data/exposure', body=body)                     
                logger.info(f'Uploaded data for indicator: {indicator} ')
            except requests.exceptions.ReadTimeout:
                logger.info(f'time out during Uploading data for indicator: {indicator} ')  
                pass            
    def uploadTyphoonData_no_event(self,json_path):
        for indicator in ["affected_population","houses_affected","forecast_severity","forecast_trigger"]: #
            try: 
                json_file_path =json_path +f'null_{indicator}' + '.json'
                with open(json_file_path) as json_file:
                    body = json.load(json_file)
                    body['date'] = self.uploadTime
                    #body['adminLevel'] = self.admin_level
                    self.apiPostRequest('admin-area-dynamic-data/exposure', body=body)                     
                logger.info(f'Uploaded data for indicator: {indicator} ')            

            except requests.exceptions.ReadTimeout:
                logger.info(f'time out during Uploading data for indicator: {indicator} ')  
                pass

    def uploadCalculatedAffected(self):
        for adminlevels in SETTINGS[self.countryCodeISO3]['levels']:#range(1,self.admin_level+1):            
            for indicator, values in self.EXPOSURE_DATA_SOURCES.items():
                if indicator == 'population':
                    with open(self.affectedFolder +
                            'affected_' + self.leadTimeLabel + '_' + self.countryCodeISO3  + '_admin_' + str(adminlevels) + '_' + 'population_affected_percentage' + '.json') as json_file:
                        body = json.load(json_file)
                        body['disasterType'] = self.getDisasterType()
                        body['date'] = self.uploadTime
                        #body['adminLevel'] = self.admin_level
                        self.apiPostRequest('admin-area-dynamic-data/exposure', body=body)
                    logger.info('Uploaded calculated_affected for indicator: ' + 'population_affected_percentage for admin level: ' + str(adminlevels))
                    with open(self.affectedFolder+'affected_' + self.leadTimeLabel + '_' + self.countryCodeISO3 + '_admin_' + str(adminlevels) + '_' + indicator + '.json') as json_file:
                        body = json.load(json_file)
                        body['disasterType'] = self.getDisasterType()
                        body['date'] = self.uploadTime
                        #body['adminLevel'] = self.admin_level
                        self.apiPostRequest('admin-area-dynamic-data/exposure', body=body)
                    logger.info(f'Uploaded calculated_affected for indicator: {indicator}' +'for admin level: ' + str(adminlevels))
                else:
                    with open(self.affectedFolder +'affected_' + self.leadTimeLabel + '_' + self.countryCodeISO3 + '_admin_' + str(adminlevels) + '_' + indicator + '.json') as json_file:
                        body = json.load(json_file)
                        body['disasterType'] = self.getDisasterType()
                        body['date'] = self.uploadTime
                        #body['adminLevel'] = self.admin_level
                        self.apiPostRequest('admin-area-dynamic-data/exposure', body=body)
                    logger.info(f'Uploaded calculated_affected for indicator: {indicator}' +'for admin level: ' + str(adminlevels))
                                    
    def uploadTrackData(self,json_path):
        json_file_path =json_path +'_tracks' + '.json'
        with open(json_file_path) as json_file:
            track_records = json.load(json_file)
        disasterType = self.getDisasterType()
        body=track_records
        body['date'] = self.uploadTime
        
        '''
        body2={}
        body2['countryCodeISO3']=body['countryCodeISO3']
        body2['leadTime']=body['leadTime']
        body2['eventName']=body['eventName']
        
        exposure=[]
        for value in body['trackpointDetails']:
            value['windspeed']=int(value['windspeed'])
            exposure.append(value)
                
        body2['trackpointDetails']=exposure
        
        '''
    
        self.apiPostRequest('typhoon-track/', body=body)
        logger.info(f'Uploaded track_data: {json_file_path}')
                    
    def uploadTrackDataAfterlandfall(self,json_path):
        json_file_path =json_path +'_tracks' + '.json'
        with open(json_file_path) as json_file:
            track_records = json.load(json_file)
        disasterType = self.getDisasterType() 
        track_records['leadTime']= '0-hour'    
        self.apiPostRequest('typhoon-track/', body=track_records)
        logger.info(f'Uploaded track_data: {json_file_path}')
        
    def uploadRasterFile(self):
        disasterType = self.getDisasterType()
        rasterFile = RASTER_OUTPUT + '0/flood_extents/flood_extent_' + self.leadTimeLabel + '_' + self.countryCodeISO3 + '.tif'
        files = {'file': open(rasterFile,'rb')}
        self.apiPostRequest('admin-area-dynamic-data/raster/' + disasterType, files=files)
        logger.info(f'Uploaded raster-file: {rasterFile}')


    def uploadTriggerPerStation(self):
        df = pd.read_json(self.triggerFolder +
                          'triggers_rp_' + self.leadTimeLabel + '_' + self.countryCodeISO3 + ".json", orient='records')
        dfStation = pd.DataFrame(index=df.index)
        dfStation['stationCode'] = df['stationCode']
        dfStation['forecastLevel'] = df['fc'].astype(np.float64,errors='ignore')
        dfStation['forecastProbability'] = df['fc_prob'].astype(np.float64,errors='ignore')
        dfStation['forecastTrigger'] = df['fc_trigger'].astype(np.int32,errors='ignore')
        dfStation['forecastReturnPeriod'] = df['fc_rp'].astype(np.int32,errors='ignore')
        dfStation['triggerLevel'] = df['triggerLevel'].astype(np.int32,errors='ignore')
        stationForecasts = json.loads(dfStation.to_json(orient='records'))
        body = {
            'countryCodeISO3': self.countryCodeISO3,
            'leadTime': self.leadTimeLabel,
            'stationForecasts': stationForecasts
        }
        self.apiPostRequest('glofas-stations/triggers', body=body)
        logger.info('Uploaded triggers per station')

    def uploadTriggersPerLeadTime(self):
        with open(self.triggerFolder +
                  'trigger_per_day_' + self.countryCodeISO3 + ".json") as json_file:
            triggers = json.load(json_file)[0]
            triggersPerLeadTime = []
            for key in triggers:
                triggersPerLeadTime.append({
                    'leadTime': str(key),
                    'triggered': triggers[key]
                })
            body = {
                'countryCodeISO3': self.countryCodeISO3,
                'triggersPerLeadTime': triggersPerLeadTime
            }
            body['disasterType'] = self.getDisasterType()
            self.apiPostRequest('event/triggers-per-leadtime', body=body)
        logger.info('Uploaded triggers per leadTime')

    def apiGetRequest(self, path, countryCodeISO3):
        from urllib3.util.retry import Retry  
        TOKEN = self.apiAuthenticate()
        
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        response = session.get(
            self.API_SERVICE_URL + path + '/' + countryCodeISO3,
            headers={'Authorization': 'Bearer ' + TOKEN}
        )

        '''
        response = requests.get(
            self.API_SERVICE_URL + path + '/' + countryCodeISO3,
            headers={'Authorization': 'Bearer ' + TOKEN}
        )
        '''
        data = response.json()
        return(data)

    def apiPostRequest(self, path, body=None, files=None):
        TOKEN = self.apiAuthenticate()
        from urllib3.util.retry import Retry

        if body != None:
            headers={'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json', 'Accept': 'application/json'}
        elif files != None:
            headers={'Authorization': 'Bearer ' + TOKEN}
        '''
        r = requests.post(
            self.API_SERVICE_URL + path,
            json=body,
            files=files,
            headers=headers
        )
        '''
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
      

        r = session.post(
            self.API_SERVICE_URL + path,
            json=body,
            files=files,
            headers=headers,
            timeout=300
        )
         
        if r.status_code >= 400:
            #logger.info(r.text)
            logger.error("PIPELINE ERROR")
            raise ValueError()


    def apiAuthenticate(self):
        API_LOGIN_URL=self.API_SERVICE_URL+'user/login'
        login_response = requests.post(API_LOGIN_URL, data=[(
            'email', self.ADMIN_LOGIN), ('password', self.ADMIN_PASSWORD)])
        return login_response.json()['user']['token']

    def getDataFromDatalake(self, filename):
        from azure.storage.blob import BlobServiceClient

        container_name='ibftyphoonforecast'
        directory_name = 'input/'
        logger.info(f'Downloading from {container_name}: /{directory_name}{filename}')

        # Create blob service client
        blob_service_client = BlobServiceClient(
            account_url=f"https://{DATALAKE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=DATALAKE_STORAGE_ACCOUNT_KEY
        )
        self._download_from_blob(blob_service_client, container_name, directory_name + filename, filename)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('./data') 


    def getDataFromDatalake2(self, datalakefolder):
        from azure.storage.blob import BlobServiceClient
  
        container_name='ibftyphoonforecast'
        directory_name = 'output/forecast'
        logger.info(f'Downloading previous model run data from {container_name}: /{directory_name}')

        # Create blob service client
        blob_service_client = BlobServiceClient(
            account_url=f"https://{DATALAKE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=DATALAKE_STORAGE_ACCOUNT_KEY
        )
    
        for layer in ["prob_within_50km","houses_affected","forecast_severity","forecast_trigger","show_admin_area","affected_population","tracks","rainfall","windspeed"]:
            logger.info(f'downloading layer {layer}')
            remote_file= f'{datalakefolder}_{layer}.json' 
            local_file_path=os.path.join(self.Output_folder,remote_file)
            self._download_from_blob(blob_service_client, container_name, remote_file, local_file_path)
 
    def _download_from_blob(self, blob_service_client, container_name, blob_name, local_file_path):
        try:
            container_client = blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob(timeout=120).readall())
        except Exception as e:
            print(e)


    def postDataToDatalake(self,datalakefolder):
        from azure.storage.blob import BlobServiceClient

        import os, uuid, sys

        container_name='ibftyphoonforecast'
        directory_name = 'output/forecast/'

        try:
            # Create blob service client
            blob_service_client = BlobServiceClient(
                account_url=f"https://{DATALAKE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
                credential=DATALAKE_STORAGE_ACCOUNT_KEY
            )
            container_client = blob_service_client.get_container_client(container_name)
            dir_client = container_client.get_directory_client(datalakefolder)
            dir_client.create_directory()
            
            for jsonfile in [x for x in os.listdir(self.Output_folder) if x.endswith('.json')]:
                local_file = open(self.Output_folder + jsonfile,'r')                
                # file_contents = local_file.read()
                # file_client = dir_client.create_file(f"{jsonfile}")
                # file_client.upload_data(file_contents, overwrite=True)
                with open(local_file, "rb") as data:
                    blob_client = container_client.get_blob_client(directory_name)
                    blob_client.upload_blob(data, overwrite=True)

        except Exception as e:
            print(e) 
    

    def postResulToDatalake(self):
        import datetime as dt
        from azure.storage.blob import BlobServiceClient     
        import time

        try:
            CONTAINER_NAME='ibftyphoonforecast'
            directory_name= 'ibf_model_results' 
            zip_filename = 'model_outputs.zip'
            filename = dt.datetime.strptime(self.uploadTime, "%Y-%m-%dT%H:%M:%SZ")
            timestamp = filename.strftime("%Y%m%dT%H")

            # Local file and destination settings
            self.zipFilesInDir(self.Output_folder, self.Output_folder + zip_filename)
            time.sleep(10) # Sleep for 10 seconds

            # Destination path inside the container (acts like folder/file.zip)
            destination_blob_path = f"{directory_name}/{timestamp}_{zip_filename}"
            destination_blob_path_ = f"{directory_name}/{zip_filename}"

            # Create blob service client
            blob_service_client = BlobServiceClient(
                account_url=f"https://{DATALAKE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
                credential=DATALAKE_STORAGE_ACCOUNT_KEY
            )

            # Get container client
            container_client = blob_service_client.get_container_client(CONTAINER_NAME)
            local_file_path= self.Output_folder + zip_filename

            # Upload the ZIP file to the specified path
            with open(local_file_path, "rb") as data:
                blob_client = container_client.get_blob_client(destination_blob_path)
                blob_client.upload_blob(data, overwrite=True)
            # Upload the ZIP file to the specified path
            with open(local_file_path, "rb") as data:
                blob_client = container_client.get_blob_client(destination_blob_path_)
                blob_client.upload_blob(data, overwrite=True)
            logger.info(f'Uploaded {zip_filename} to {destination_blob_path} in container {CONTAINER_NAME}')
            
            return 1
        
        except Exception as e:
            print(e)
            return 0
            
            
    def postResulToSkype(self,skypUsername,skypPassword,channel_id):
        from skpy import Skype        
        msg='AUTOMATED MESSAGE FROM DATAPIPELINE- Model output files based on latest ECMWF forecast can be found here:- https://510ibfsystem.blob.core.windows.net/ibftyphoonforecast/ibf_model_results/model_outputs.zip '
        sk = Skype(skypUsername,skypPassword)
        channel = sk.chats.chat(channel_id) 
        channel.sendMsg(msg)
               
    def zipFilesInDir(self,dirName, zipFileName):
        from zipfile import ZipFile
        import os
        from os.path import basename       
        #files = [ fi for fi in os.listdir(dirName) if not fi.endswith(".json") ]
        files_to_zip = [fi for fi in os.listdir(dirName) if not (fi.endswith(".json") or fi.endswith(".zip"))]
        with ZipFile(zipFileName, 'w') as zipObj: # create a ZipFile object            
            for filename in files_to_zip:#os.listdir(dirName): # Iterate over all the files in directory
                filePath = os.path.join(dirName, filename)
                zipObj.write(filePath, arcname=filename)#                zipObj.write(filePath, basename(filePath))
    
    def uploadImage(self,typhoons,eventName='no-name'):
        disasterType = self.getDisasterType()
       
        imageFile = self.Output_folder + self.countryCodeISO3 + '_' + typhoons +'_houseing_damage.png'
  
        files = {
            'image': (imageFile, open(imageFile, 'rb'), "image/png"), 
            }
        data = {"submit": "Upload Image" }
        
        path_=f'event/event-map-image/{self.countryCodeISO3}/{disasterType}/{eventName}'          
        self.apiPostRequestImage(path_,
                                 files=files,
                                 data=data
                                 )
        logger.info(f'Uploaded image-file: {imageFile}')
        
    def apiPostRequestImage(self, path,files=None,data=None):
        TOKEN = self.apiAuthenticate()

        headers={'Authorization': 'Bearer ' + TOKEN}

        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
      

        r = session.post(
            self.API_SERVICE_URL + path,  
            files=files,
            data=data,
            headers=headers
        )
         
        if r.status_code >= 400:
            #logger.info(r.text)
            logger.error("PIPELINE ERROR")
            raise ValueError()
import os
from datetime import datetime, timedelta
##################
## LOAD SECRETS ##
##################

# 1. Try to load secrets from Azure key vault (i.e. when running through Logic App) if user has access
try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
    az_credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    secret_client = SecretClient(vault_url='https://ibf-flood-keys.vault.azure.net', credential=az_credential)
    
    AZURE_STORAGE_ACCOUNT=secret_client.get_secret("AZURE-STORAGE-ACCOUNT").value
    AZURE_CONNECTING_STRING=secret_client.get_secret("AZURE-CONNECTING-STRING").value
    
    ADMIN_LOGIN = secret_client.get_secret("IBF-TEST-LOGIN").value
    PHP_PASSWORD=secret_client.get_secret("IBF-TEST-PASSWORD").value
    UCL_USERNAME=secret_client.get_secret("UCL-USERNAME").value
    UCL_PASSWORD=secret_client.get_secret("UCL-PASSWORD").value  
    DATALAKE_STORAGE_ACCOUNT_NAME = secret_client.get_secret("DATALAKE-STORAGE-ACCOUNT-NAME").value
    DATALAKE_STORAGE_ACCOUNT_KEY = secret_client.get_secret("DATALAKE-STORAGE-ACCOUNT-KEY").value
    DATALAKE_API_VERSION = '2018-11-09'
 


except Exception as e:
    print('No access to Azure Key vault, skipping.')

#2. Try to load secrets from env-variables (i.e. when using Github Actions)
try:
    import os
    
    ADMIN_LOGIN = os.environ['ADMIN_LOGIN']
    IBF_URL=os.environ['IBF_API_URL']
    #PHP_PASSWORD=os.environ['IBF_PASSWORD']
    DATALAKE_STORAGE_ACCOUNT_NAME = os.environ['DATALAKE_STORAGE_ACCOUNT_NAME']        
    DATALAKE_STORAGE_ACCOUNT_KEY_ = os.environ["DATALAKE_STORAGE_ACCOUNT_KEY"]
    #DATALAKE_STORAGE_ACCOUNT_KEY_ =os.environ.get("DATALAKE-STORAGE-ACCOUNT-KEY")
    print('Environment variables found.')
    DATALAKE_STORAGE_ACCOUNT_KEY=f'{DATALAKE_STORAGE_ACCOUNT_KEY_}=='
    
    DATALAKE_API_VERSION = '2018-11-09'
except Exception as e:
    print('No environment variables found.')

# 3. If 1. and 2. both fail, then assume secrets are loaded via secrets.py file (when running locally). If neither of the 3 options apply, this script will fail.
try:
    from typhoonmodel.utility_fun.secrets import *
except ImportError:
    print('No secrets file found.')



countryCodes=['PHL']

# COUNTRY SETTINGS
SETTINGS_SECRET = {
    "PHL": {
        "IBF_API_URL":IBF_API_URL,#'https://ibf-demo.510.global/api/',
        "ADMIN_LOGIN": ADMIN_LOGIN,
        "ADMIN_PASSWORD": ADMIN_PASSWORD,
        "UCL_USERNAME": UCL_USERNAME,
        "UCL_PASSWORD": UCL_PASSWORD,
        "AZURE_STORAGE_ACCOUNT": AZURE_STORAGE_ACCOUNT,
        "AZURE_CONNECTING_STRING": AZURE_CONNECTING_STRING,
        "admin_level": 3,
        "mock": False,
        "mock_nontrigger_typhoon_event": 'nontrigger_scenario',
        "if_mock_trigger": True,
        "mock_trigger_typhoon_event": 'trigger_scenario',
        "notify_email": True
    },
}

###################
## PATH SETTINGS ##
###################

start_time = datetime.now()
MAIN_DIRECTORY='/home/fbf/'

#MAIN_DIRECTORY='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/IBF_TYPHOON_DATA_PIPELINE/IBF-Typhoon-model/'

ADMIN_PATH =MAIN_DIRECTORY+'data/gis_data/phl_admin3_simpl2.geojson'
PRE_DISASTER_INDICATORS = MAIN_DIRECTORY+'data/pre_disaster_indicators/all_predisaster_indicators.csv'
CENTROIDS_PATH = MAIN_DIRECTORY+'data/gis_data/centroids_windfield.geojson'
 
ecmwf_remote_directory='20220925060000'#None#'20220923060000'#(start_time - timedelta(hours=24)).strftime("%Y%m%d120000")
High_resoluation_only_Switch=False
#ecmwf_remote_directory=None#(start_time - timedelta(hours=10)).strftime("%Y%m%d000000")#None#'20220714120000'
typhoon_event_name=None
ECMWF_CORRECTION_FACTOR=1.6
ECMWF_LATENCY_LEADTIME_CORRECTION=10 #
Active_Typhoon_event_list=['NORU']#'MA-ON']
WIND_SPEED_THRESHOLD=20



Alternative_data_point = (start_time - timedelta(hours=24)).strftime("%Y%m%d")  
data_point = start_time.strftime("%Y%m%d")      

Input_folder = MAIN_DIRECTORY+ 'forecast/Input/'
Output_folder = MAIN_DIRECTORY+ 'forecast/Output/'
ECMWF_folder = MAIN_DIRECTORY+'forecast/Input/ECMWF/'
rainfall_path = MAIN_DIRECTORY+'forecast/Input/rainfall/'
mock_data_path = MAIN_DIRECTORY+'data/mock/'
ML_model_input = MAIN_DIRECTORY+'data/model_input/df_modelinput_july.csv'
if not os.path.exists(Input_folder):
    os.makedirs(Input_folder)
if not os.path.exists(Output_folder):
    os.makedirs(Output_folder)
if not os.path.exists(ECMWF_folder):
    os.makedirs(ECMWF_folder)
if not os.path.exists(rainfall_path):
    os.makedirs(rainfall_path)  

Population_Growth_factor=1.15 #(1+0.02)^7 adust 2015 census data by 2%growth for the pst 7 years 

Housing_unit_correction={'year':['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022'],
                        'facor':[0.88,0.89,0.91,0.92,0.93,0.95,0.96,0.97,0.99,1.00,1.01,1.03,1.04,1.06,1.07,1.09,1.10]}
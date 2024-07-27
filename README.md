
# Table of contents

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Typhoon Impact forecasting model">Typhoon Impact forecasting model</a>
    </li>
	    <li>
      <a href="#Installation">Installation</a>
    </li>
    <li>
      <a href="#Running With Docker">Running pipeline With Docker</a>
      <ul>
        <li><a href="#Build Container">Build and Run Container</a></li>
     </ul>
    </li>
    <li><a href="#Acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- Typhoon Impact forecasting model -->
## Typhoon Impact forecasting model

This tool was developed as a trigger mechanism for the typhoon Early action protocol of the Philippines Red Cross FbF project. The model will predict the potential damage of a typhoon before landfall, and the prediction will be percentage of completely damaged houses per municipality. The tool is available under the [GPL license](https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model/blob/master/LICENSE)

To run the pipeline, you need access to an Data.zip, and credentiials for 510 Datalake and FTP server. If you or your organization is interested in using the pipeline, 
please contact [510 Global](https://www.510.global/contact-us/) to obtain the credentials. You will receive a file called `secrets`, which you need to place in the top-level directory.

The main use of this data pipeline is to update the status of Typhoon IBF portal based on the latest ECMWF forecast. The status of Triggered/Not Triggred is defined based on EAP trigger value. Currently this is based on Average impact. This value can be updated the setting.py file [at this line ](https://github.com/rodekruis/IBF_TYPHOON_DATA_PIPELINE/blob/master/IBF-Typhoon-model/src/typhoonmodel/utility_fun/settings.py#L92) Defult value is Average, other possible values are 50,70 and 90 , which is the percentage of ensamble members passing the treshold.

<!-- Installation -->
## Installation

1. Clone the repo
2. Change `/IBF-Typhoon-model/src/typhoonmodel/utility_fun/secrets.py.template` to `secrets.py` and fill in the necessary passwords.
3. Install Docker-compose [follow this link](https://docs.docker.com/desktop/windows/install/)

<!-- Running Pipeline With Docker -->
## Running With Docker on local machine 

You will need to have `docker` and `docker-compose` installed.


<!-- Build and Run Container -->
### Build Container

To build and run the image, ensure you are in the top-level directory and execute:
```
docker-compose up --build


```
When you are finished the run 
```
docker-compose down
```
to remove any docker container(s).

## Running with Docker on Azure logicapp

Follow the instraction [here](https://docs.google.com/document/d/10E1BhPu55tjaPbSSACRQ0Ot2-8K-neAlPY5ex390Vu4/edit) there 
is also a workflow in github action to update docker image in azure registery, which will be the image used in logic app.   

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- [Germen Red Cross](https://www.drk.de/en/)
- [Philippines Red Cross](https://redcross.org.ph/)
- [Rode Kruis](https://www.rodekruis.nl/)

# DESRGAN


## Introduction
This work is motivated by a real-world challenge of providing high-resolution and skilful daily weather forecasts for upcoming months, which are of huge value to climate-sensitive sectors. Such Seasonal Climate Forecasts (SCF) are routinely provided by General Circulation Models (GCM), but need downscaling techniques to increase spatial resolution and consistency with local observations. Current operational downscaling techniques, such as Quantile Mapping (QM), learn pre-specified relationships between local observations and GCM hindcasts to provide SCFs with limited improvement. Promising downscaling techniques based on Single Image Super-Resolution (SISR) have been developed but only on simplified situations. We formulate the downscaling daily rainfall ensemble forecast problem as an optimisation problem and provide SISR models as candidates. According to characteristics of daily rainfall forecsts, we develop 8 times SISR techniques based on Generative Adversarial Network (GAN) by including outstanding SISR building blocks. A model called Downscaling with Enhanced Super-Resolution GAN (DESRGAN) is finalised using an ensemble forecast skill metric Continuous Ranked Probability Score (CRPS), and tested on downscaling 60km ACCESS-S1 rainfall forecasts to 5km with lead time up to 216 days in Australia. Testing results on three representative years illustrate DESRGAN has higher forecast accuracy and skills, measured by mean absolute error and CRPS respectively, than bicubic interpolation, QM, another SISR-based downscaling technique VDSR, and a long-standing SCF benchmark, climatology. This is the first time that, as far as we know, downscaled SCFs outperform the benchmark in general, and demonstrates the potential of deep-learning for SCF downscaling operation after more development. 

![DESRGAN](/img/DESRGANstructure.png)


## Train
### Training data 
####Low resolution
In 2017, the Australian Meteorology Bureau announced the next generation access series GCM, which was later installed on supercomputers in the office in 2018. A worldwide combined model, the seasonal climate and earth system Simulator (ACCESS-S), is based on the UK's global combined seasonal prediction system glosea5-gc2. The ACCESS-S contains 11 different ensemble members for seasonal forecasting and leading 217 days due to disruptions and improved ensemble technology,  including ten disturbed members and oneunperturbed centre member. 
####High resolution
The Australian Water Availability Project (AWAP) is to monitor the state and trend of the terrestrial water balance of the Australian continent, using model-data fusion methods to combine both measurements and modelling. The project determines the past history and present state of soil moisture and all water fluxes contributing to changes in soil moisture (rainfall, transpiration, soil evaporation, surface runoff and deep drainage), across the entire Australian continent at a spatial resolution of 5 km. Using the same basic framework, the project provides soil moistures and water fluxes over the Australian continent in three forms: (1) weekly/monthly near-real-time reporting, (2) monthly/annual historical time series from 1900, and (3) monthly climatologies (on request).

In the training set, we used 60km Raw atmosphere grid ACCESS-S precipitation data as input and 5km AWAP data as the target.
All the data were stored on the project named [ub7](http://www.bom.gov.au/metadata/catalogue/19115/ANZCW0503900697/) and [rr8](http://www.bom.gov.au/metadata/catalogue/19115/ANZCW0503900567/), you need to require permission from [NCI](https://nci.org.au/) on the paths (g/data/ub7/) and (g/data/rr8/) respectively.

### Pre-processing data
[todo]
We cropped the data to the same geographic region, with longitude and latitude ranges are 111.975E to 156.275E and 44.525S to 9.975S, respectively. Go to the data_processing folder and type the following commands, and the data will be cropped and relocated:

 ```bash
    python3 /pre_processing/access_pre.py
    python3 /pre_processing/awap_pre.py
 ```

### Begin to train
[todo]

```bash
    python train.py 
```

### Begin to evaluation

```bash
    python eval_DESRGAN.py 
```


## Results
### CRPS comparison
| ![space-1.png](/img/CRPS_SS_2012-1.png) | 
|:--:| 
| *Average CRPS Skill Scores across Australia for forecasts made in 2012* |

### MAE comparison
| ![space-1.png](/img/MAE_2012-1.png) | 
|:--:| 
| *Average MAE across Australia for daily precipitation forecasts made on 48 different initialisation dates in 2012* |

### Visual Results

| ![Watch the video](/img/visual.gif) | 
|:--:| 
| *Example of ensemble_1 comparison* |


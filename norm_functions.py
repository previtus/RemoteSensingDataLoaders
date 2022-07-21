import numpy as np

# Sentinel-2 L1C

BANDS_S2_BRIEF = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]
RESCALE_PARAMS = {
    "B1" : {  "x0": 7.3,
              "x1": 7.6,
              "y0": -1,
              "y1": 1,
    },
    "B2" : {  "x0": 6.9,
              "x1": 7.5,
              "y0": -1,
              "y1": 1,
    },
    "B3" : {  "x0": 6.5,
              "x1": 7.4,
              "y0": -1,
              "y1": 1,
    },
    "B4" : {  "x0": 6.2,
              "x1": 7.5,
              "y0": -1,
              "y1": 1,
    },
    "B5" : {  "x0": 6.1,
              "x1": 7.5,
              "y0": -1,
              "y1": 1,
    },
    "B6" : {  "x0": 6.5,
              "x1": 8,
              "y0": -1,
              "y1": 1,
    },
    "B7" : {  "x0": 6.5,
              "x1": 8,
              "y0": -1,
              "y1": 1,
    },
    "B8" : {  "x0": 6.5,
              "x1": 8,
              "y0": -1,
              "y1": 1,
    },
    "B8A" : { "x0": 6.5,
              "x1": 8,
              "y0": -1,
              "y1": 1,
    },
    "B9" : {  "x0": 6,
              "x1": 7,
              "y0": -1,
              "y1": 1,
    },
    "B10" : { "x0": 2.5,
              "x1": 4.5,
              "y0": -1,
              "y1": 1,
    },
    "B11" : { "x0": 6,
              "x1": 8,
              "y0": -1,
              "y1": 1,
    },
    "B12" : { "x0": 6,
              "x1": 8,
              "y0": -1,
              "y1": 1,
    }
}

def norm_S2_log_minmax(data):
    bands = data.shape[0] # for example 15  
    for band_i in range(bands):
        data_one_band = data[band_i,:,:]
        if band_i < len(BANDS_S2_BRIEF):
            # log
            data_one_band = np.log(data_one_band)
            data_one_band[np.isinf(data_one_band)] = np.nan

            # rescale
            r = RESCALE_PARAMS[BANDS_S2_BRIEF[band_i]]
            x0,x1,y0,y1 = r["x0"], r["x1"], r["y0"], r["y1"] 
            data_one_band = ((data_one_band - x0) / (x1 - x0)) * (y1 - y0) + y0
        data[band_i,:,:] = data_one_band
    return data


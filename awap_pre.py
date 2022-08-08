import xarray as xr
import numpy as np
import cv2
import numpy.ma as ma
import os

# mask = np.load("/scratch/iu60/rw6151/qm_mask.npy")

for file in (os.listdir("/g/data/rr8/OBS/AWAP_ongoing/v0.6/grid_05/daily/precip/")[:10]):
    # open file
    if file.startswith("precip_total") and "1990" <= file[18:22] <= "2012":
        ds_raw = xr.open_dataset("/g/data/rr8/OBS/AWAP_ongoing/v0.6/grid_05/daily/precip/" + file)
        # no nan in awap_total
        # ds_raw = ds_raw.fillna(0)

        lon = ds_raw["lon"].values
        lat = ds_raw["lat"].values

        for i in ds_raw['time'].values[:]:
            ds_selected_domained = ds_raw.sel(time=i)['precip']
            # awap mask
            # ds_selected_domained = ma.array(ds_selected_domained, mask=mask)
            da_interp = xr.DataArray(ds_selected_domained, dims=("lat", "lon"), coords={"lat": lat, "lon": lon, "time": i}, name='precip')
            expanded_da = xr.concat([da_interp], "time")
            # save to netcdf
            expanded_da.to_netcdf("/scratch/iu60/rw6151/awap_split_total_masked/" + str(i)[:4] + str(i)[5:7]+ str(i)[8:10] +".nc")
        ds_raw.close()

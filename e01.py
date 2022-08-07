import xarray as xr
import numpy as np
import cv2
import numpy.ma as ma
import os
import sys

from mpl_toolkits.basemap import maskoceans

# mask = np.load("/scratch/iu60/rw6151/awap_mask_40.npy")

exx = sys.argv[0][-6:-3]
print(exx)

for file in (os.listdir("/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/" + exx + "/")):
    # open file
    if file.startswith("da_pr_2002") or file.startswith("da_pr_2007"):
        ds_raw = xr.open_dataset(
            "/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/" + exx + "/" + file)
        ds_raw = ds_raw.fillna(0)
        da_selected = ds_raw.isel(time=0)['pr']

        lon = ds_raw["lon"].values
        lat = ds_raw["lat"].values
        a = np.logical_and(lon >= 111.975, lon <= 156.275)
        b = np.logical_and(lat >= -44.525, lat <= -9.975)

        da_selected_au = da_selected[b, :][:, a].copy()

        # resize lat & lon
        n = 1.5  # 60km -> 40km   == 1.5 scale

        size = (int(da_selected_au.lon.size * n), int(da_selected_au.lat.size * n))

        new_lon = np.linspace(
            da_selected_au.lon[0], da_selected_au.lon[-1], size[0])
        new_lon = np.float32(new_lon)
        new_lat = np.linspace(
            da_selected_au.lat[0], da_selected_au.lat[-1], size[1])
        new_lat = np.float32(new_lat)

        # interp and merge
        i = ds_raw['time'].values[0]
        da_selected = ds_raw.sel(time=i)['pr']
        da_selected_au = da_selected[b, :][:, a].copy()
        temp = cv2.resize(da_selected_au.values, size,
                          interpolation=cv2.INTER_CUBIC)
        temp = np.clip(temp, 0, None)
        temp = cv2.resize(temp, size, interpolation=cv2.INTER_CUBIC)
        # mask
        lons_mask, lats_mask = np.meshgrid(new_lon, new_lat)
        temp = maskoceans(lons_mask, lats_mask, temp, resolution='c', grid=1.25)

        da_interp = xr.DataArray(temp, dims=("lat", "lon"),
                                 coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
        ds_total = xr.concat([da_interp], "time")

        for i in ds_raw['time'].values[:]:
            ds_selected_domained = ds_raw.sel(time=i)['pr']
            da_selected_au = ds_selected_domained[b, :][:, a].copy()
            temp = cv2.resize(da_selected_au.values, size,
                              interpolation=cv2.INTER_CUBIC)
            temp = np.clip(temp, 0, None)
            # awap mask
            lons_mask, lats_mask = np.meshgrid(new_lon, new_lat)
            temp = maskoceans(lons_mask, lats_mask, temp, resolution='c', grid=1.25)

            da_interp = xr.DataArray(temp, dims=("lat", "lon"),
                                     coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
            expanded_da = xr.concat([da_interp], "time")
            ds_total = xr.merge([ds_total, expanded_da])

        # save to netcdf
        ds_total.to_netcdf(
            "/scratch/iu60/rw6151/BI/" + exx + "/" + file)
        ds_raw.close()

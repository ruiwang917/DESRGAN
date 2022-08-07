import RRDBNet_arch as arch
import argparse
import glob
from os import mkdir
from os.path import isdir

import cv2
import numpy as np
import torch
import xarray as xr
from torch.autograd import Variable
from tqdm import tqdm

import torch.nn as nn


def test(batch_input):
    # divide and conquer strategy due to GPU memory limit

    # _, H, W = batch_input.size()

    slice_output = model_G(batch_input)

    slice_output = slice_output.cpu().data.numpy()

    slice_output = np.clip(slice_output[0], 0., 1.)

    slice_output = cv2.resize(np.squeeze(slice_output), (886, 691),
                              interpolation=cv2.INTER_CUBIC)

    slice_output = np.clip(slice_output, 0, 1)

    return slice_output


parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('-n', type=int, default=2,
                    help='Divisor, make larger when GPU memory shortage')
args = parser.parse_args()

# Model
for model in ['model_G_i000004.pth']:

    model_G = arch.RRDBNetx4x2(1, 1, 64, 23, gc=32).cuda()
    model_G.load_state_dict(torch.load(
        './PESR/checkpoint/vtest/' + model).module.state_dict(), strict=True)
    if torch.cuda.device_count() > 1:
        print("!!!Let's use " + str(torch.cuda.device_count()) + "GPUs!")
        model_G = nn.DataParallel(model_G, range(torch.cuda.device_count()))
    model_G.eval()

    # Test
    with torch.no_grad():
        # Input nc file
        # exx = ['e01', 'e02', 'e03', 'e04', 'e05', 'e06', 'e07', 'e08', 'e09', 'e10', 'e11']
        # ['e05', 'e06', 'e08']
        exx = ['e05', 'e06', 'e08']
        path = '/scratch/iu60/rw6151/access_40_217_masked/e*/*.nc'
        allfiles = glob.glob(path)

        for en in exx:
            files = []

            for i in allfiles:
                if "da_pr_2009" in i or "da_pr_2011" in i:
                    if en in i:
                        files.append(i)

            files.sort()

            for fn in files:
                ds_raw = xr.open_dataset(fn) * 86400
                ds_raw = ds_raw.fillna(0)
                ds_raw = np.clip(ds_raw, 0, 1000)
                ds_raw = np.log1p(ds_raw) / 7
                startdate = str(ds_raw['time'].values[0])[:10]
                # To 0-1
                da_selected = ds_raw.isel(time=0)["pr"]

                lon = ds_raw["lon"].values
                lat = ds_raw["lat"].values
                a = np.logical_and(lon >= 111.975, lon <= 156.275)
                b = np.logical_and(lat >= -44.525, lat <= -9.975)
                da_selected_au = da_selected[b, :][:, a].copy()

                n = 1
                # lat691 lon886
                size = (int(886), int(691))
                # size = (int(632), int(728))
                new_lon = np.linspace(
                    da_selected_au.lon[0], da_selected_au.lon[-1], size[0])
                new_lon = np.float32(new_lon)
                new_lat = np.linspace(
                    da_selected_au.lat[0], da_selected_au.lat[-1], size[1])
                new_lat = np.float32(new_lat)

                da_selected = da_selected.values
                da_selected_au = cv2.resize(
                    da_selected, (110, 86), interpolation=cv2.INTER_CUBIC)

                da_selected_au = np.asarray(da_selected_au).astype(np.float32)
                da_selected_au = da_selected_au[np.newaxis, np.newaxis, ...]
                batch_input = Variable(torch.from_numpy(da_selected_au)).cuda()

                # Save to file
                i = ds_raw['time'].values[0]
                batch_output = test(batch_input)
                da_interp = xr.DataArray(np.expm1(batch_output * 7), dims=("lat", "lon"),
                                         coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
                ds_total = xr.concat([da_interp], "time")

                for i in ds_raw['time'].values[1:217]:
                    ds_selected_domained = ds_raw.sel(time=i)['pr'].values
                    da_selected_au = ds_selected_domained[b, :][:, a].copy()
                    da_selected_au = cv2.resize(
                        da_selected_au, (110, 86), interpolation=cv2.INTER_CUBIC)

                    da_selected_au = np.asarray(da_selected_au).astype(np.float32)

                    da_selected_au = da_selected_au[np.newaxis, np.newaxis, ...]
                    batch_input = Variable(torch.from_numpy(da_selected_au)).cuda()

                    batch_output = test(batch_input)
                    da_interp = xr.DataArray(np.expm1(batch_output * 7), dims=("lat", "lon"),
                                             coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
                    expanded_da = xr.concat([da_interp], "time")
                    ds_total = xr.merge([ds_total, expanded_da])

                if not isdir(model[:15]):
                    mkdir('./' + model[:15])

                if not isdir('./' + model[:15] + '/' + en + "/"):
                    mkdir('./' + model[:15] + '/' + en)

                savepath = "./" + model[:15] + "/" + \
                           en + "/" + startdate + "_" + en + ".nc"
                ds_total.to_netcdf(savepath)

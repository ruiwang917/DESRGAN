import data_processing_tool as dpt
import random
from torch.utils.data import Dataset
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import platform
from datetime import timedelta, date, datetime
import numpy as np
import os
import time
import properscoring as ps


def mae(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens - hr)).sum(axis=0) / ens.shape[0]


def mae_mean(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens.mean(axis=0) - hr))


def mae_median(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((np.median(ens, axis=0) - hr))


def bias(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return (ens - hr).sum(axis=0) / ens.shape[0]


def bias_median(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.median(ens, axis=0) - hr


def rmse(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.sqrt(((ens - hr) ** 2).sum(axis=(0)) / ens.shape[0])


# ===========================================================
# Training settings
# ===========================================================


class ACCESS_AWAP_cali(Dataset):
    '''

2.using my net to train one channel to one channel.

    '''

    def __init__(self, start_date=date(1990, 1, 1), end_date=date(1990, 12, 31), regin="AUS", lr_transform=None,
                 hr_transform=None, shuffle=True, args=None):
        #         print("=> BARRA_R & ACCESS_S1 loading")
        #         print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        # '/g/data/rr8/OBS/AWAP_ongoing/v0.6/grid_05/daily/precip/'
        self.file_AWAP_dir = args.file_AWAP_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args = args

        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date

        self.regin = regin
        self.leading_time_we_use = args.leading_time_we_use

        self.ensemble_access = ['e01', 'e02', 'e03', 'e04',
                                'e05', 'e06', 'e07', 'e08', 'e09', 'e10', 'e11']
        self.ensemble = []
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])

        self.dates = self.date_range(start_date, end_date)

        self.filename_list = self.get_filename_with_time_order(
            self.file_ACCESS_dir)
        if not os.path.exists(self.file_ACCESS_dir):
            print(self.file_ACCESS_dir)
            print("no file or no permission")

        en, cali_date, date_for_AWAP, time_leading = self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)

        data_awap = dpt.read_awap_data_fc_get_lat_lon(
            self.file_AWAP_dir, date_for_AWAP)
        self.lat_awap = data_awap[1]
        self.lon_awap = data_awap[2]

    def __len__(self):
        return len(self.filename_list)

    def date_range(self, start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    def get_filename_with_no_time_order(self, rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(rootdir, list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:] == ".nc":
                    _files.append(path)
        return _files

    def get_filename_with_time_order(self, rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for date in self.dates:
            for i in range(self.leading_time_we_use, self.leading_time_we_use + 1):

                for en in self.ensemble:
                    access_path = rootdir + en + "/" + \
                                  date.strftime("%Y-%m-%d") + "_" + en + ".nc"
                    #                   print(access_path)
                    if os.path.exists(access_path):

                        if date == self.end_date and i == 1:
                            break
                        path = []
                        path.append(en)
                        awap_date = date + timedelta(i)
                        path.append(date)
                        path.append(awap_date)
                        path.append(i)
                        _files.append(path)

        # 最后去掉第一行，然后shuffle
        return _files

    def mapping(self, X, min_val=0., max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        # 将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b - a) / (Xmax - Xmin) * (X - Xmin)
        return Y

    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t = time.time()

        # read_data filemame[idx]
        en, access_date, awap_date, time_leading = self.filename_list[idx]

        lr = dpt.read_access_data_calibrataion(
            self.file_ACCESS_dir, en, access_date, time_leading, "pr")
        label = dpt.read_awap_data_fc(self.file_AWAP_dir, awap_date)

        return np.array(lr), np.array(label), torch.tensor(
            int(en[1:])), torch.tensor(int(access_date.strftime("%Y%m%d"))), torch.tensor(time_leading)


def write_log(log, args):
    print(log)
    if not os.path.exists("./save/" + args.train_name + "/"):
        os.mkdir("./save/" + args.train_name + "/")
    my_log_file = open("./save/" + args.train_name + '/train.txt', 'a')
    my_log_file.write(log + '\n')
    my_log_file.close()
    return


def main(year, days):
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=0,
                        help='number of threads for data loading')

    parser.add_argument('--cpu', action='store_true', help='cpu only?')

    # hyper-parameters
    parser.add_argument('--train_name', type=str,
                        default="cali_crps", help='training name')

    parser.add_argument('--batch_size', type=int,
                        default=44, help='training batch size')
    parser.add_argument('--testBatchSize', type=int,
                        default=4, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=200,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning Rate. Default=0.01')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed to use. Default=123')

    # model configuration
    parser.add_argument('--upscale_factor', '-uf', type=int,
                        default=4, help="super resolution upscale factor")
    parser.add_argument('--model', '-m', type=str, default='vdsr',
                        help='choose which model is going to use')

    # data
    parser.add_argument('--pr', type=bool, default=True, help='add-on pr?')

    parser.add_argument('--train_start_time', type=type(datetime(1990,
                                                                 1, 25)), default=datetime(1990, 1, 2), help='r?')
    parser.add_argument('--train_end_time', type=type(datetime(1990,
                                                               1, 25)), default=datetime(1990, 2, 9), help='?')
    parser.add_argument('--test_start_time', type=type(datetime(2012,
                                                                1, 1)), default=datetime(2012, 1, 1), help='a?')
    parser.add_argument('--test_end_time', type=type(datetime(2012,
                                                              12, 31)), default=datetime(2012, 12, 31), help='')

    parser.add_argument('--dem', action='store_true', help='add-on dem?')
    parser.add_argument('--psl', action='store_true', help='add-on psl?')
    parser.add_argument('--zg', action='store_true', help='add-on zg?')
    parser.add_argument('--tasmax', action='store_true', help='add-on tasmax?')
    parser.add_argument('--tasmin', action='store_true', help='add-on tasmin?')
    parser.add_argument('--leading_time_we_use', type=int,
                        default=1, help='add-on tasmin?')
    parser.add_argument('--ensemble', type=int, default=11,
                        help='total ensambles is 11')
    parser.add_argument('--channels', type=float, default=0,
                        help='channel of data_input must')
    # [111.85, 155.875, -44.35, -9.975]
    parser.add_argument('--domain', type=list,
                        default=[111.975, 156.275, -44.525, -9.975], help='dataset directory')

    parser.add_argument('--file_ACCESS_dir', type=str,
                        default="/scratch/iu60/rw6151/Large_Patch/40+log_v3/model_G_i000002/",
                        help='dataset directory')
    parser.add_argument('--file_AWAP_dir', type=str, default="/scratch/iu60/rw6151/Split_AWAP_masked_total/",
                        help='dataset directory')

    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half', 'double'),
                        help='FP precision for test (single | half)')

    args = parser.parse_args()

    sys = platform.system()
    args.dem = False
    args.train_name = "pr_DESRGAN"
    args.channels = 0
    if args.pr:
        args.channels += 1
    if args.zg:
        args.channels += 1
    if args.psl:
        args.channels += 1
    if args.tasmax:
        args.channels += 1
    if args.tasmin:
        args.channels += 1
    if args.dem:
        args.channels += 1
    print("training statistics:")
    print("  ------------------------------")
    print("  trainning name  |  %s" % args.train_name)
    print("  ------------------------------")
    print("  num of channels | %5d" % args.channels)
    print("  ------------------------------")
    print("  num of threads  | %5d" % args.n_threads)
    print("  ------------------------------")
    print("  batch_size     | %5d" % args.batch_size)
    print("  ------------------------------")
    print("  using cpu only | %5d" % args.cpu)

    lr_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    hr_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    args.test_start_time = datetime(year, 1, 1)
    args.test_end_time = datetime(year, 12, 31)

    write_log("start", args)

    for lead in range(0, days):
        args.leading_time_we_use = lead

        data_set = ACCESS_AWAP_cali(args.test_start_time, args.test_end_time, lr_transform=lr_transforms,
                                    hr_transform=hr_transforms, shuffle=False, args=args)

        test_data = DataLoader(data_set,
                               batch_size=args.batch_size,
                               shuffle=False,
                               num_workers=args.n_threads, drop_last=False)

        mean_mae_model = []
        mean_mae_mean_model = []
        mean_mae_mediam_model = []
        mean_bias_model = []
        mean_rmse_model = []
        mean_crps_model = []
        mean_bias_median_model = []

        for batch, (pr, hr, _, _, _) in enumerate(test_data):

            with torch.set_grad_enabled(False):

                sr_np = pr.cpu().numpy()
                hr_np = hr.cpu().numpy()

                for i in range(args.batch_size // args.ensemble):
                    a = np.squeeze(
                        sr_np[i * args.ensemble:(i + 1) * args.ensemble])
                    b = np.squeeze(hr_np[i * args.ensemble])
                    mae_DESRGAN = mae(a, b)
                    mae_mean_DESRGAN = mae_mean(a, b)
                    mae_median_DESRGAN = mae_median(a, b)
                    bias_DESRGAN = bias(a, b)
                    bias_median_DESRGAN = bias_median(a, b)
                    rmse_DESRGAN = rmse(a, b)
                    skil_DESRGAN = ps.crps_ensemble(b, np.transpose(a, (1, 2, 0)))

                    mean_mae_model.append(mae_DESRGAN)
                    mean_mae_mean_model.append(mae_mean_DESRGAN)
                    mean_mae_mediam_model.append(mae_median_DESRGAN)
                    mean_bias_model.append(bias_DESRGAN)
                    mean_bias_median_model.append(bias_median_DESRGAN)
                    mean_rmse_model.append(rmse_DESRGAN)
                    mean_crps_model.append(skil_DESRGAN)

        if not os.path.exists("/scratch/iu60/rw6151/new_crps/save/mae/v3_2/" + str(year)):
            os.mkdir("/scratch/iu60/rw6151/new_crps/save/mae/v3_2/" + str(year))
        np.save("/scratch/iu60/rw6151/new_crps/save/mae/v3_2/" + str(year) + "/lead_time" + str(lead) + '_whole',
                np.mean(mean_mae_model, axis=0))

        if not os.path.exists("/scratch/iu60/rw6151/new_crps/save/mae_mean/v3_2/" + str(year)):
            os.mkdir("/scratch/iu60/rw6151/new_crps/save/mae_mean/v3_2/" + str(year))
        np.save(
            "/scratch/iu60/rw6151/new_crps/save/mae_mean/v3_2/" + str(year) + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_mae_mean_model, axis=0))

        if not os.path.exists("/scratch/iu60/rw6151/new_crps/save/mae_median/v3_2/" + str(year)):
            os.mkdir("/scratch/iu60/rw6151/new_crps/save/mae_median/v3_2/" + str(year))
        np.save(
            "/scratch/iu60/rw6151/new_crps/save/mae_median/v3_2/" + str(year) + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_mae_mediam_model, axis=0))

        if not os.path.exists("/scratch/iu60/rw6151/new_crps/save/bias/v3_2/" + str(year)):
            os.mkdir("/scratch/iu60/rw6151/new_crps/save/bias/v3_2/" + str(year))
        np.save("/scratch/iu60/rw6151/new_crps/save/bias/v3_2/" + str(year) + "/lead_time" + str(lead) + '_whole',
                np.mean(mean_bias_model, axis=0))

        if not os.path.exists("/scratch/iu60/rw6151/new_crps/save/bias_median/v3_2/" + str(year)):
            os.mkdir("/scratch/iu60/rw6151/new_crps/save/bias_median/v3_2/" + str(year))
        np.save(
            "/scratch/iu60/rw6151/new_crps/save/bias_median/v3_2/" + str(year) + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_bias_median_model, axis=0))

        if not os.path.exists("/scratch/iu60/rw6151/new_crps/save/rmse/v3_2/" + str(year)):
            os.mkdir("/scratch/iu60/rw6151/new_crps/save/rmse/v3_2/" + str(year))
        np.save("/scratch/iu60/rw6151/new_crps/save/rmse/v3_2/" + str(year) + "/lead_time" + str(lead) + '_whole',
                np.mean(mean_rmse_model, axis=0))

        if not os.path.exists("/scratch/iu60/rw6151/new_crps/save/crps_ss/v3_2/" + str(year)):
            os.mkdir("/scratch/iu60/rw6151/new_crps/save/crps_ss/v3_2/" + str(year))
        np.save("/scratch/iu60/rw6151/new_crps/save/crps_ss/v3_2/" + str(year) + "/lead_time" + str(lead) + '_whole',
                np.mean(mean_crps_model, axis=0))


if __name__ == '__main__':
    main(year=2002, days=30)
    print('2002done')
    main(year=2007, days=30)
    print('2007done')
    main(year=2012, days=7)
    print('2012done')

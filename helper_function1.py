import pandas as pd
import seaborn as sns
import numpy as np
import func_helpers as fh
import cyclic_helper as ch
import plot_helper as ph
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from scipy import signal
from datetime import datetime, timedelta
import copy
from PIL import Image, ImageDraw


df_list = []
df_new_length = []
tag_list = []
Xpos_list = []
Ypos_list = []
loc_times_list = []
MaxHeight_list = []
MaxWidth_list = []
time_pos_dict_list = []

R765RF_D5 = {'4':[1,2,3,4,5,6,7,8,9,10,11,12],'6':[1,2],'7':[1,2,3],'10':[1,2,3,4,5,6,7],'13': [1,2,3],'15':[1,2,3,4,5]}

R781_D2 = {'2':[1,2,3,4],'3':[1,2,3,4,5,6,7,8,9],'5':[1,2,3,4,5,6], '9':[1,2,3,4]}
R781_D3 = {'2':[1,2,3,4,5,6],'3':[1,2,3,4,5,6,7,8],'5':[1,2,3,4,5,6,7,8],'9':[1,2],'11':[1,2],'14':[1,2]}
R781_D4 = {'2':[1,2,3,4,5,6,7,8,9],'3':[1,2,3,4,5,6,7,8],'5':[1,2,3,4]} #LM max < 0.40

R808_D1 = {'1':[1,2],'6':[1,2],'9':[1],'13':[1],'14':[1],'15':[1,2,3]}
R808_D6 = {'9':[1,2,3,4],'12':[1,2,3,4,5,6,7,8,9],'13':[1,2,3,4,5]}
R808_D7 = {'12':[1,2,3,4,5,6,7,8,9],'15':[1,2,3,4]}


R859_D1 = {'1':[1,2,3,4,5,6],'3':[1,2,3,4,5,6,7],'4':[1,2,3,4],'5':[1,2],'6':[1,2,3,4,5,6],'7':[1,2,3,4,5,6,7,8],'8':[1,2,3,4,5,6,7],'14':[1,2,3,4,5,6,7,8,9]}
R859_D2 = {'1':[1,2,3,4,5],'6':[1,2,3,4,5,6,7,8],'7_0001':[1,2,3,4,5,6,7,8],'8':[1,2,3,4,5],'10':[1,2,3,4,5,6],'11':[1,2,3,4,5,6,7,8,9,10],'12':[1,2,3,4],'13':[1,2,3,4,5],'14':[1,2,3,4,5,6,7,8]}
R859_D3 = {'1_0001':[1,2,3,4,5],'3':[1,2,3,4,5,6],'6':[1,2,3,4,5,6,7,8,9],'7_0001':[1,2,3,4,5,6,7,8],'8_0001':[1,2,3,4],'10':[1,2,3,4,5,6,7,8],'12':[1,2,3,4]}

R886_D1 = {'2':[1,2],'5':[1,2,3],'9':[1,2,3,4,5],'10':[1,2]}
R886_D2 = {'2':[1,2],'5':[1,2,3],'10':[1,2,3],'12':[1,2,3,4,5]}
R886_D3 = {'4':[1,2]}


def make_tuple(a, b):
    # make any object a and b to a tuple (a,b)
    return (a, b)

#######################

def xdy_ydx(pair):
    """Calculate the area integral between a pair of timeseries data."""
    x, y = ch.match_ends(ch.tv_norm(ch.mean_center(np.nan_to_num(np.asarray(pair)))))
    # x, y = ch.tv_norm(ch.mean_center(np.nan_to_num(np.asarray(pair))))
    return (x.dot(ch.cyc_diff(y)) - y.dot(ch.cyc_diff(x))) / 2

def create_spike_convo(loc_times_list_1,MaxHeight_list_1):
    loc_times_arr_1 = (np.array(loc_times_list_1) // 1e5).astype(np.int) ### unit in 1/10s
    MaxHeight_arr_1 = np.array(MaxHeight_list_1)
    min_1 = loc_times_arr_1[0]
    max_1 = loc_times_arr_1[-1]
    loc_times_arr_1 = loc_times_arr_1 - min_1
    res_1 = np.zeros(max_1-min_1+1)
    for i in range(len(MaxHeight_arr_1)):
        res_1[loc_times_arr_1[i]] = MaxHeight_arr_1[i]
    tmp_1 = (np.arange(min_1*1e5, min_1*1e5+len(res_1)*1e5, 1e5)).astype(np.int)

    # print("---")
    # print(min_1, min_2)
    return res_1, tmp_1
def create_spike_convo2(loc_times_list_1,MaxHeight_list_1,loc_times_list_2,MaxHeight_list_2):
        loc_times_arr_1 = (np.array(loc_times_list_1) // 1e5).astype(np.int) ### unit in 1/10s
        MaxHeight_arr_1 = np.array(MaxHeight_list_1)
        min_1 = loc_times_arr_1[0]
        max_1 = loc_times_arr_1[-1]
        loc_times_arr_1 = loc_times_arr_1 - min_1
        res_1 = np.zeros(max_1-min_1+1)
        for i in range(len(MaxHeight_arr_1)):
            res_1[loc_times_arr_1[i]] = MaxHeight_arr_1[i]
        tmp_1 = (np.arange(min_1*1e5, min_1*1e5+len(res_1)*1e5, 1e5)).astype(np.int)

        loc_times_arr_2 = (np.array(loc_times_list_2) // 1e5).astype(np.int) ### unit in 1/10s
        MaxHeight_arr_2 = np.array(MaxHeight_list_2)
        min_2 = loc_times_arr_2[0]
        max_2 = loc_times_arr_2[-1]
        loc_times_arr_2 = loc_times_arr_2 - min_2
        res_2 = np.zeros(max_2-min_2+1)
        for i in range(len(MaxHeight_arr_2)):
            res_2[loc_times_arr_2[i]] = MaxHeight_arr_2[i]
        tmp_2 = (np.arange(min_2*1e5, min_2*1e5+len(res_2)*1e5, 1e5)).astype(np.int)
        # print("---")
        # print(min_1, min_2)
        min_ = min(min_1, min_2)
        max_ = max(max_1, max_2)
        idx_1 = min_1 - min_
        idx_2 = min_2 - min_
        new_1 = np.zeros(max_-min_+1)
        new_2 = np.zeros(max_-min_+1)
        new_1[idx_1:idx_1+len(res_1)] = res_1
        new_2[idx_2:idx_2+len(res_2)] = res_2
        tmp = (np.arange(min_*1e5, min_*1e5+len(new_2)*1e5, 1e5)).astype(np.int)

        # return res_1, tmp_1, res_2, tmp_2
        return new_1, tmp, new_2, tmp


a = 1.5
b = 3
def continuous_function(x):
    return (np.exp(-x/b) * x**(a-1))

def cumul_area_iter(x, y):
    x = tuple(map(float, x))
    y = tuple(map(float, y))
    # x, y = ch.match_ends(ch.tv_norm(np.nan_to_num(np.asarray(tuple((x, y))))))
    x, y = ch.tv_norm(np.nan_to_num(np.asarray(tuple((x, y)))))
    assert len(x) == len(y)
    z = [0] * len(x)
    for n in range(1, len(z)):
        z[n] = z[n-1] + (x[0]-x[n])*(y[n-1]-y[n]) - (y[0]-y[n])*(x[n-1]-x[n])
    return [val/2 for val in z]

def get_xy_tuples(xindex, yindex):
    xtuples = df_list[xindex][0]
    ytuples = df_list[yindex][0]
    return xtuples, ytuples


def normalize_timestamp(timestamp_tuple):
    time_difference_minutes = []
    starttime = timestamp_tuple[0] / 6e7
    # starttime = datetime.fromtimestamp(starttime)

    for milliseconds in timestamp_tuple:
        seconds = milliseconds / 6e7
        
        # timestamp_datetime = datetime.fromtimestamp(seconds)
        # time_difference = timestamp_datetime - starttime
        time_difference = seconds - starttime

        # print(time_difference)

        # time_difference_minutes.append(time_difference.total_seconds())
        time_difference_minutes.append(time_difference)


    # print("Timestamp as datetime:", time_difference_minutes)
    return time_difference_minutes

def get_time_axis_min(xids, yids):
    time_axis_end = min(xids[-1],yids[-1])
    if len(xids) > len(yids):
        return xids, time_axis_end
    return yids, time_axis_end

def location_plot(Xpos, Ypos, loc_times, XorY):
# Example location data with corresponding times
    latitudes = Ypos
    longitudes = Xpos
    times = loc_times  # Example time values (can be in hours)

    # Create a colormap
    cmap = ListedColormap(['blue', 'green', 'yellow', 'red'])
    normalize = Normalize(vmin=min(times), vmax=max(times))

    # Create a scatter plot of locations with colors based on time
    # plt.figure(figsize=(10, 6))
    if XorY:
        plt.subplot(2,4,3)
    else:
        plt.subplot(2,4,7)
    plt.scatter(longitudes, latitudes, c=times, cmap=cmap, marker='o', s=10, norm=normalize)

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Time')


    # Set labels and title
    plt.xlabel('Xposition')
    plt.ylabel('Yposition')
    plt.title('Location Plot')

    # Show the plot
    plt.grid(True)

def get_raw_timestamp(start_min, end_min, xids):
        raw_timestamp = []
        startid = xids[0]
        start = (start_min * 6e7) + startid
        end = (end_min * 6e7) + startid
        for ids in xids:
            if ids > start and ids < end:
                raw_timestamp.append(ids)
        return raw_timestamp

def get_value_tolist(dict, timelist):
        value = []
        for ids in timelist:
            value.append(dict.get(ids))
        return value

def get_adjacent_time_differences(lst):
    differences = [0]
    for i in range(len(lst) - 1):
        difference = lst[i + 1] - lst[i]
        differences.append(difference//1e6)
    return differences


def get_idx(lst, target):
    for i in range(len(lst)):
        if lst[i]>target:
            return i
    return None

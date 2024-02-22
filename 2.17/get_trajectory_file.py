import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import func_helpers as fh
import cyclic_helper as ch
import plot_helper as ph
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
    loc_times_arr_1 = (np.array(loc_times_list_1) // 1e5).astype(np.int64) ### unit in 1/10s
    MaxHeight_arr_1 = np.array(MaxHeight_list_1)
    min_1 = loc_times_arr_1[0]
    max_1 = loc_times_arr_1[-1]
    loc_times_arr_1 = loc_times_arr_1 - min_1
    res_1 = np.zeros(max_1-min_1+1)
    for i in range(len(MaxHeight_arr_1)):
        res_1[loc_times_arr_1[i]] = MaxHeight_arr_1[i]
    tmp_1 = (np.arange(min_1*1e5, min_1*1e5+len(res_1)*1e5, 1e5)).astype(np.int64)

    # print("---")
    # print(min_1, min_2)
    return res_1, tmp_1
def create_spike_convo2(loc_times_list_1,MaxHeight_list_1,loc_times_list_2,MaxHeight_list_2):
        loc_times_arr_1 = (np.array(loc_times_list_1) // 1e5).astype(np.int64) ### unit in 1/10s
        MaxHeight_arr_1 = np.array(MaxHeight_list_1)
        min_1 = loc_times_arr_1[0]
        max_1 = loc_times_arr_1[-1]
        loc_times_arr_1 = loc_times_arr_1 - min_1
        res_1 = np.zeros(max_1-min_1+1)
        for i in range(len(MaxHeight_arr_1)):
            res_1[loc_times_arr_1[i]] = MaxHeight_arr_1[i]
        tmp_1 = (np.arange(min_1*1e5, min_1*1e5+len(res_1)*1e5, 1e5)).astype(np.int64)

        loc_times_arr_2 = (np.array(loc_times_list_2) // 1e5).astype(np.int64) ### unit in 1/10s
        MaxHeight_arr_2 = np.array(MaxHeight_list_2)
        min_2 = loc_times_arr_2[0]
        max_2 = loc_times_arr_2[-1]
        loc_times_arr_2 = loc_times_arr_2 - min_2
        res_2 = np.zeros(max_2-min_2+1)
        for i in range(len(MaxHeight_arr_2)):
            res_2[loc_times_arr_2[i]] = MaxHeight_arr_2[i]
        tmp_2 = (np.arange(min_2*1e5, min_2*1e5+len(res_2)*1e5, 1e5)).astype(np.int64)
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
        tmp = (np.arange(min_*1e5, min_*1e5+len(new_2)*1e5, 1e5)).astype(np.int64)

        # return res_1, tmp_1, res_2, tmp_2
        return new_1, tmp, new_2, tmp


########################################
########################################
a = 1.5
b = 0.015
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

def normalize_timestamp_second(timestamp_tuple, reference):
    time_difference_minutes = []
    starttime = reference / 1e6
    # starttime = datetime.fromtimestamp(starttime)

    for milliseconds in timestamp_tuple:
        seconds = milliseconds / 1e6
        
        # timestamp_datetime = datetime.fromtimestamp(seconds)
        # time_difference = timestamp_datetime - starttime
        time_difference = seconds - starttime

        # print(time_difference)

        # time_difference_minutes.append(time_difference.total_seconds())
        time_difference_minutes.append(time_difference)


    # print("Timestamp as datetime:", time_difference_minutes)
    return time_difference_minutes

def normalize_timestamp(timestamp_tuple, reference):
    time_difference_minutes = []
    starttime = reference / 6e7
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
        return xids, time_axis_end, 0
    return yids, time_axis_end, 1

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
        if lst[i]>=target:
            return i
    return None



######################
######################

spike_convo_list = []
time_spike_list = []

for j in R765RF_D5:
    # print(j)
    for i in R765RF_D5[j]:
        # print(i)
        df = pd.read_csv("/Users/eeevashen/Downloads/city_data/R765RF_D5/TT"+ j + "/cl-maze1."+str(i)+".csv")
        maze_name = " R765RF_D5/TT"+ j + "/cl-maze1."+str(i)
        tag_list.append(maze_name)
        select_cols = df.columns[-5:]
        df.drop(df.index[0:4], inplace=True)
        df_new = df[select_cols].astype(int)

        Xpos = df_new['XPos'].tolist()
        Ypos = df_new['YPos'].tolist()
        loc_times = df_new['Timestamp'].tolist()
        MaxHeight = df_new['MaxHeight'].tolist()
        MaxWidth = df_new['MaxWidth'].tolist()
        df_new['XYpos'] = df[['XPos', 'YPos']].apply(lambda x: tuple(x), axis=1)
        time_pos_dict = df_new.set_index('Timestamp')['XYpos'].to_dict()

        # print(time_pos_dict)

        # loc_times = [df_new['Timestamp'].tolist()[0]] + df_new['Timestamp'].tolist()

        df_new['Firingtime'] = range(1,len(df)+1)
        data = [make_tuple(df_new['Timestamp'].tolist(), df_new['Firingtime'].tolist())]
        Xpos_list.append(Xpos)
        Ypos_list.append(Ypos)
        loc_times_list.append(loc_times)
        df_new_length.append(df.shape[0])
        MaxHeight_list.append(MaxHeight)
        MaxWidth_list.append(MaxWidth)
        time_pos_dict_list.append(time_pos_dict)

    
        spike_len1 = len(MaxHeight)
        t_continuous1 = np.linspace(0, 20, 100)
        norm_conv1 = continuous_function(t_continuous1)/np.max(continuous_function(t_continuous1))

        # plt.figure()
        # plt.plot(MaxHeight)
        # plt.show()

        spike_convo, time_spike = create_spike_convo(loc_times,MaxHeight)
        time_spike1 = list(time_spike)
        time_spike = [time_spike[0]]+ list(time_spike)


        # print("spike_convo length:" + str(len(spike_convo)))
        # print("normalized continuous function length:" + str(len(norm_conv1)))


        convolution_result1 = signal.convolve(spike_convo, norm_conv1, mode='same') ## adding 0 to the unspike points
        convolution_result1 = [0] + list(convolution_result1)



        spike = [make_tuple(time_spike, convolution_result1)]
        df_list.append(spike)
        
    
        time_spike_list.append(time_spike1)
        spike_convo_list.append(spike_convo)



#every code above should be sealed in to a new file
        
ret, rate_list, xids, yids = fh.make_lead_matrix(fh.flatten(df_list), intfunc=xdy_ydx)
LM, phases, perm, sortedLM, evals = ret



## which paris need to be analysis
########################################
pairs = (3, 4)


xtuples, ytuples = get_xy_tuples(pairs[0], pairs[1])
(xids, xvals), (yids, yvals) = fh.prune(xtuples, ytuples, ids=True)

xids1 = xids
yids1 = yids

# get_time_axis_min is finding the bigger id between xids and yids
time_ids, time_axis_end, flag = get_time_axis_min(xids, yids)
if flag == 0:
    reference = xids1[0]
else:
    reference = yids1[0]

# raw_ids, raw_end = get_time_axis_min(xids, yids)

xvals = tuple(map(float, xvals))
yvals = tuple(map(float, yvals))

xid_sec = normalize_timestamp_second(xids,reference)
yid_sec = normalize_timestamp_second(yids,reference)
if len(xid_sec) > len(yid_sec):
    time_id_sec = yid_sec
else:
    time_id_sec = xid_sec

xids = normalize_timestamp(xids, reference)
yids = normalize_timestamp(yids, reference)


# areavals = cumul_area_iter(xvals, yvals)
# if flag == 0:
#     time_ids, time_axis_end = xids, xids[-1]
# else:
#     time_ids, time_axis_end = yids, yids[-1]


max_col_value = max(max(xvals), max(yvals))

# # Plot the original list, continuous function, and convolution result
# plt.figure(figsize=(10, 5))
# plt.figure()


# plt.figure(figsize=(12, 6))
# plt.plot(time_ids,areavals)
# plt.xticks(np.arange(0, max(time_ids)+3, 1)) 
# plt.rcParams['xtick.labelsize'] = 6  # Set the font size of x-axis tick labels
# plt.rcParams['font.family'] = 'serif' 
# plt.xlabel('Timestamps') 
# plt.ylabel('CumlArea') 
# plt.title('Cumulated Area Plot')

# plt.show()


def zoom_in_convo(start_time_min, end_time_min, start_reference, xids, yids, xvals, yvals):

    def normalize_timestamp_path(timestamp_tuple, start_reference):
        time_difference_minutes = []
        starttime = start_reference / 6e7
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

    def get_idx(lst, target):
        for i in range(len(lst)):
            if lst[i]>target:
                return i
        return None
    print(len(xids))

    smaller_idx = get_idx(xids, start_time_min)
    larger_idx = get_idx(xids, end_time_min)
    print(smaller_idx, larger_idx)


    
    xvals_zoom = xvals[smaller_idx:larger_idx]
    yvals_zoom= yvals[smaller_idx:larger_idx]
    xids_zoom = xids[smaller_idx:larger_idx]
    yids_zoom = yids[smaller_idx:larger_idx]


    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1) 
    plt.plot(xids_zoom, xvals_zoom)
    plt.xticks(np.arange(start_time_min, end_time_min+1, 0.1)) 
    plt.ylim(0,max_col_value)
    plt.title("convolution plot for" + str(pairs[0]))
    plt.xlabel('Timestamps in mins') 
    plt.ylabel('MaxHeight*continuous function sampe') ## need to normalized

    plt.subplot(2, 1, 2) 
    plt.plot(yids_zoom,yvals_zoom)
    plt.xticks(np.arange(start_time_min, end_time_min+1, 0.1)) 
    plt.ylim(0,max_col_value)
    plt.title("convolution plot for" + str(pairs[1]))
    plt.xlabel('Timestamps in mins') 
    plt.ylabel('MaxHeight*continuous function sampe') ## need to normalized
    plt.show()




start_reference = reference #the reference is the min timestamp of paris in spike data

def get_position_plot(start_time_min, end_time_min, start_reference, mice_data_file):

    df = pd.read_csv(mice_data_file,low_memory=False)
    new_columns = ['Timestamp_raw', 'Xpos_raw','Ypos_raw','Head_angle_raw']
    df.columns = new_columns
    df['Timestamp_raw'] = df.iloc[:,0]
    df['Xpos_raw'] = df.iloc[:,1]
    df['Ypos_raw'] = df.iloc[:,2]
    df['Head_angle_raw'] = df.iloc[:,3]
    # print(df)

    ## since we use the pop.s.ascii data, we may not need to delete the (0,0)
    df_cleaned = df[df['Xpos_raw'] != 0.0]
    df_cleaned = df_cleaned[df_cleaned['Ypos_raw'] != 0.0]
    df_cleaned = df_cleaned[df_cleaned['Xpos_raw'] != -99.0]
    df_cleaned = df_cleaned[df_cleaned['Ypos_raw'] != -99.0]
    df_cleaned = df_cleaned[df_cleaned['Xpos_raw'] <= 610.0]

    df_test = df[df['Xpos_raw'] == 0.0]

    timestamp = df_cleaned['Timestamp_raw'].tolist()
    Xpos = df_cleaned['Xpos_raw'].tolist()
    Ypos = df_cleaned['Ypos_raw'].tolist()



    def normalize_timestamp_path(timestamp_tuple, start_reference):
        time_difference_minutes = []
        starttime = start_reference / 6e7
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

    def get_idx(lst, target):
        for i in range(len(lst)):
            if lst[i]>target:
                return i
        return None
    
    start_reference

    smaller_idx = get_idx(normalize_timestamp_path(timestamp,start_reference), start_time_min)
    larger_idx = get_idx(normalize_timestamp_path(timestamp,start_reference), end_time_min)


    
    x_coordinates = Xpos[smaller_idx:larger_idx]
    y_coordinates = Ypos[smaller_idx:larger_idx]
    time = normalize_timestamp_path(timestamp[smaller_idx:larger_idx],start_reference)


    print("===========")
    # Saving the lists to a text file
    with open('/Users/eeevashen/Desktop/summer_reserach /trajectory_R765RF' + str(start_time_min) +'-' + str(end_time_min) +'.txt', 'w') as file:
        # Writing each list on a separate line
        # file.write("timerange" + str(start_time_min)+ " to " + str(end_time_min))
        file.write("time: " + ', '.join(map(str, time)) + '\n')
        file.write("xpos: " + ', '.join(map(str, x_coordinates)) + '\n')
        file.write("ypos: " + ', '.join(map(str, y_coordinates)) + '\n')
        # file.write("===========================================")

    
# zoom_in_convo(21.7, 22.3, start_reference, xids, yids, xvals, yvals)
get_position_plot(21.7, 22.3, start_reference, "/Users/eeevashen/Downloads/city_data/R765RF_D5/Pos.p.ascii.csv")


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


        print("spike_convo length:" + str(len(spike_convo)))
        print("normalized continuous function length:" + str(len(norm_conv1)))


        convolution_result1 = signal.convolve(spike_convo, norm_conv1, mode='same') ## adding 0 to the unspike points
        convolution_result1 = [0] + list(convolution_result1)



        spike = [make_tuple(time_spike, convolution_result1)]
        df_list.append(spike)
        
    
        time_spike_list.append(time_spike1)
        spike_convo_list.append(spike_convo)


# print("-----------------------")
# print(df_list[0])
# print("-----------------------")


ret, rate_list, xids, yids = fh.make_lead_matrix(fh.flatten(df_list), intfunc=xdy_ydx)
LM, phases, perm, sortedLM, evals = ret

index_list = []
LM_copy = copy.deepcopy(LM)
LM_copy_flatten = LM_copy.flatten()
top_indices = np.argsort(LM_copy_flatten)[-5:]
row_indices, col_indices = np.unravel_index(top_indices, LM_copy.shape)
print("Indices of the 5 largest numbers:")
for i in range(5):
    index_list.append((row_indices[4-i], col_indices[4-i]))
    # print(f"Number: {LM_copy[row_indices[i], col_indices[i]]}, Index: ({row_indices[i]}, {col_indices[i]})")
    print(f"Number: {LM_copy[row_indices[4-i], col_indices[4-i]]}, Index: ({row_indices[4-i]}, {col_indices[4-i]}), Corresponding maze: {tag_list[row_indices[4-i]], tag_list[col_indices[4-i]]}")

print("=======================")
print(index_list)
m = np.argmax(LM_copy_flatten)
r, c = divmod(m, LM.shape[1])
print(r, c)

print("=======================")
for i in range(len(df_list)):
    print(f"Index: {i}, Corresponding maze: {tag_list[i]}")


# Function defined in plot_helpers.py
fig = plt.figure(figsize=(12,8))
idxs = ph.plot_initial(ret, fig=fig, labels=tag_list)
fig

plt.show()


i = 0
for pairs in index_list:
    # commented
    # print(pairs)
    # print(type(pairs))


    # xtuples = (time_spike, convolution_result1)
   
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

    # subplot1
    plt.figure(figsize=(16,8))
    plt.subplot(2,4,1)
    
    
    max_yvalues = max(max(spike_convo_list[pairs[0]]),max(spike_convo_list[pairs[1]]))

    plt.vlines(normalize_timestamp(time_spike_list[pairs[0]],reference), ymin=0, ymax=spike_convo_list[pairs[0]], color='red', linestyle='dashed')
    # plt.vlines(normalize_timestamp(loc_times_list[pairs[0]], reference), ymin=0, ymax=MaxHeight_list[pairs[0]], color='red', linestyle='dashed')
    plt.xlabel('Timestamps(min)') 
    plt.ylabel('MaxHeight') 
    plt.ylim(0, max_yvalues)
    plt.title('Timeseries Plot for' + str(pairs[0]))

    # plt.show()

    # subplot2
    plt.subplot(2,4,5)

    plt.vlines(normalize_timestamp(time_spike_list[pairs[1]],reference), ymin=0, ymax=spike_convo_list[pairs[1]], color='red', linestyle='dashed')
    # plt.vlines(normalize_timestamp(loc_times_list[pairs[1]], reference), ymin=0, ymax=MaxHeight_list[pairs[1]], color='blue', linestyle='dashed')
    plt.xlabel('Timestamps(min)') 
    plt.ylabel('MaxHeight') 
    plt.ylim(0, max_yvalues)
    plt.title('Timeseries Plot for' + str(pairs[1]))
    # plt.show()


    max_col_value = max(max(xvals), max(yvals))

    # Plot the original list, continuous function, and convolution result
    # plt.figure(figsize=(10, 5))
    # plt.figure()
    plt.subplot(2,4,2)
    plt.plot(xids, xvals)
    #plt.xlim(0,time_spike1[-1]/600)
    plt.ylim(0,max_col_value)
    plt.title("convolution plot for" + str(pairs[0]))
    plt.xlabel('Timestamps in mins') 
    plt.ylabel('MaxHeight*continuous function sampe') ## need to normalized
    # plt.show()

    # plt.figure()
    plt.subplot(2,4,6)
    plt.plot(yids,yvals)
    # plt.xlim(0,time_spike2[-1]/600)
    plt.ylim(0,max_col_value)
    plt.title("convolution plot for" + str(pairs[1]))
    plt.xlabel('Timestamps in mins') 
    plt.ylabel('MaxHeight*continuous function sampe') ## need to normalized
    # plt.show()


    
    areavals = cumul_area_iter(xvals, yvals)
    if flag == 0:
        time_ids, time_axis_end = xids, xids[-1]
    else:
        time_ids, time_axis_end = yids, yids[-1]

    # print("======")
    # print(time_ids)
      # subplot3
    plt.subplot(2,4,4)
    plt.plot(time_ids,areavals)
    
    plt.xlabel('Timestamps') 
    plt.ylabel('CumlArea') 
    plt.title('Cumulated Area Plot')

    location_plot(Xpos_list[pairs[0]], Ypos_list[pairs[0]], normalize_timestamp(loc_times_list[pairs[0]], reference), 1)
    location_plot(Xpos_list[pairs[1]], Ypos_list[pairs[1]], normalize_timestamp(loc_times_list[pairs[1]], reference), 0)
    plt.show()


####try to write a function to find prpper smaller_minutes and larger_minutes

    def get_next_index(cur_idx, x, step):
        n = len(x)
        cur_val = x[cur_idx]
        for i in range(cur_idx, n):
            if cur_val + step < x[i]:
                return i
        return n


    def detect_jump(x, step, y, threshold):
        res = []
        left_idx = 0
        right_idx = get_next_index(left_idx, x, step)

        tmp_res = None
        while right_idx != len(x):
            left_y = y[left_idx]
            right_y = y[right_idx]
            if right_y - left_y >= threshold:
                if tmp_res is None:
                    tmp_res = [left_idx, right_idx]
                else:
                    tmp_res[1] = right_idx
            else:
                if tmp_res != None:
                    res.append(tmp_res)
                tmp_res = None
            left_idx = right_idx
            right_idx = get_next_index(right_idx, x, step)


        return res
    

    print("=================")
    print("lenth of areavals" + str(len(areavals)))
    print('lenth of xvals' + str(len(xvals)))
    print('lenth of time_id_sec' + str(len(time_id_sec)))
    threshold = 0.05 * abs(max(areavals) - min(areavals))
    time_range_idx = detect_jump(time_id_sec, 25, areavals, threshold)
    time_range = []
    for each in time_range_idx:
        time_range.append((time_id_sec[each[0]], time_id_sec[each[1]]))
    


    # def detect_change(x, delta_x, threshold, time_range):
    #     smaller_minutes = x - delta_x
    #     larger_minutes = x + delta_x
    #     smaller_idx = get_idx(time_ids, smaller_minutes)
    #     mid_idx = get_idx(time_ids, x)
    #     larger_idx = get_idx(time_ids, larger_minutes)
    #     y_left = areavals[smaller_idx]
    #     y_mid = areavals[mid_idx]
    #     y_right = areavals[larger_idx]
    #     if abs(y_right - y_left) > threshold:
    #         tmp = make_tuple(x - delta_x, x + delta_x)
    #         # print(tmp)
    #         time_range.append(tmp) 
    #     return time_range

    # time_range = []
    # for mins in range(1, int(max(time_ids))-1):
    #     time_range = detect_change(mins, 1, 1, time_range)

    # print("====the time range need to observe==========")
    # print(time_range)

    # def detect_jump(x, delta_x, threshold, time_range):
    #     smaller_minutes = x - delta_x
    #     larger_minutes = x + delta_x
    #     smaller_idx = get_idx(time_ids, smaller_minutes)
    #     larger_idx = get_idx(time_ids, larger_minutes)
    #     y = {}
    #     for idx in range(smaller_idx, larger_idx):
    #         y[areavals[idx]] = idx
    #         # y.append(areavals[idx])
    #     sort_y = dict(sorted(y.items())) ## sorted by key(areavalue)
    #     # print(sort_y.keys())
    #     if max(sort_y.keys()) - min(sort_y.keys()) > threshold:
    #         t1 = sort_y[min(sort_y.keys())]
    #         t2 = sort_y[max(sort_y.keys())]
    #         tmp = make_tuple(t1 - 1/12, t2)
    #         time_range.append(tmp) 
    #     return time_range
    
    # time_range = []
    # threshold = 0.25 * abs(max(areavals) - min(areavals))
    # for mins in range(1, int(max(time_ids))-1):
    #     # time_range = detect_jump(mins, 0.5, threshold, time_range)
    #     detect_change(mins, 0.35, threshold, time_range)

    # print("====the time range need to observe==========")
    # print(time_range)

    print(len(time_spike_list))
    print(len(spike_convo_list))
   
    # def continuous_function(x):
    #     b = 0.015
    #     a = 1.5
    #     return (np.exp(-x/b) * x**(a-1))

    # fig_debug = plt.figure(figsize=(16,8))
    # plt.subplot(2,1,1)
    # a = spike_convo_list[pairs[0]]

    # # new_a = []
    # # for i in range(len(a)):
    # #     new_a.append(a[i])
    # #     for j in range(500):
    # #         new_a.append(0)
            
    # # array_length = 376
    # # a = np.random.rand(array_length)
    # # # a = [[i, 0] for i in (a)]

    # # new_array = []
    # # for i in range(array_length):
    # #     new_array.append(a[i])
    # #     for j in range(500):
    # #         new_array.append(0)
    # # a = new_array
    # #plt.plot(a)
    # plt.vlines(normalize_timestamp(time_spike_list[pairs[0]],reference), ymin=0, ymax=a, color='red', linestyle='dashed')
    # plt.xlabel('Timestamps(min)') 
    # plt.ylabel('MaxHeight') 
    # # plt.ylim(0, max_yvalues)
    # plt.title('Timeseries Plot for' + str(pairs[0]))

    # plt.subplot(2,1,2)
    
    # #plt.xlim(0,time_spike1[-1]/600)

    # spike_len1 = 100
    # t_continuous1 = np.linspace(0, 20, spike_len1)
    # norm_conv1 = continuous_function(t_continuous1)/np.max(continuous_function(t_continuous1))
    # convolution_result1 = signal.convolve(a, norm_conv1, mode='same')
    # plt.plot(convolution_result1)

    # # plt.ylim(0,max_col_value)
    # plt.title("convolution plot for" + str(pairs[0]))
    # plt.xlabel('Timestamps in mins') 
    # plt.ylabel('MaxHeight*continuous function sampe') ## need to normalized
    # plt.show()








    # for pair in time_range:

    #     smaller_sec = pair[0]
    #     larger_sec = pair[1]

    #     ## get correspinding index in areaval of each time_id
    #     smaller_idx = get_idx(time_id_sec, smaller_sec)
    #     larger_idx = get_idx(time_id_sec, larger_sec)

    #     smaller_idx_scatter_1 = get_idx(normalize_timestamp_second(loc_times_list[pairs[0]], reference), smaller_sec)
    #     larger_idx_scatter_1 = get_idx(normalize_timestamp_second(loc_times_list[pairs[0]], reference), larger_sec)
    #     smaller_idx_scatter_2 = get_idx(normalize_timestamp_second(loc_times_list[pairs[1]], reference), smaller_sec)
    #     larger_idx_scatter_2 = get_idx(normalize_timestamp_second(loc_times_list[pairs[1]], reference), larger_sec)
    #     print("--------------")
    #     #print(normalize_timestamp_second(loc_times_list[pairs[0]], reference))
    #     print(smaller_sec)
        





    #     # smaller_minutes = pair[0]
    #     # larger_minutes = pair[1]
    #     # ## get correspinding index in areaval of each time_id
    #     # smaller_idx = get_idx(time_ids, smaller_minutes)
    #     # larger_idx = get_idx(time_ids, larger_minutes)

    #     # smaller_idx_scatter_1 = get_idx(normalize_timestamp(loc_times_list[pairs[0]], reference ), smaller_minutes)
    #     # larger_idx_scatter_1 = get_idx(normalize_timestamp(loc_times_list[pairs[0]], reference), larger_minutes)
    #     # smaller_idx_scatter_2 = get_idx(normalize_timestamp(loc_times_list[pairs[1]], reference), smaller_minutes)
    #     # larger_idx_scatter_2 = get_idx(normalize_timestamp(loc_times_list[pairs[1]], reference), larger_minutes)


    #     # print(smaller_idx_scatter)
    #     # print(Xpos_list[pairs[0]][smaller_idx_scatter])
        
    #     fig = plt.figure()
    #     #plt.figure(figsize=(16,8))
    #     fig = plt.figure()
    #     fig.set_figheight(8)
    #     fig.set_figwidth(16)
    #     spec = gridspec.GridSpec(ncols=2, nrows=3, width_ratios=[1, 1],  height_ratios=[10, 3, 3], wspace=0.3, hspace=0.3)
        

    #     ax0 = plt.subplot(spec[0])
    #     # plt.subplot(2,2,1)
    #     ax0.plot(time_ids,areavals, color="blue")
    #     ax0.plot(time_ids[smaller_idx: larger_idx],areavals[smaller_idx: larger_idx], color="red")

    #     ax0.set_xlabel('Timestamps', fontsize=6) 
    #     ax0.set_ylabel('CumlArea', fontsize=6) 
    #     ax0.set_title('Cumulated Area Plot', fontsize=8)

    #     import pop_s_2 as ps2

    #     print("mazepaires:" + str(pairs))
    #     print("timerange:" + str(pair))

    #     if flag == 0:
    #         path_start_raw = xids1[0]
    #     else:
    #         path_start_raw = yids1[0]

        
        
    #     # plt.subplot(2,2,2)
    #     ax1= plt.subplot(spec[1])
    #     ps2.get_position_plot_sec(ax1, smaller_sec, larger_sec, path_start_raw, "/Users/eeevashen/Downloads/city_data/R765RF_D5/Pos.p.ascii.csv")
    #     ax1.scatter(Xpos_list[pairs[0]][smaller_idx_scatter_1:larger_idx_scatter_1], Ypos_list[pairs[0]][smaller_idx_scatter_1:larger_idx_scatter_1], marker='o', color = "red")
    #     ax1.scatter(Xpos_list[pairs[1]][smaller_idx_scatter_2:larger_idx_scatter_2], Ypos_list[pairs[1]][smaller_idx_scatter_2:larger_idx_scatter_2], marker='o', color = "blue")
    #     ax1.legend(("mouse path in jump peroid","green - path start", "purple - path end", "red scatter points - leading", "blue scatter points - following"), loc = "upper right", fontsize = 'x-small')
    #     # plt.legend("blue scatter points - following")
    #     # plt.legend("green - start")
    #     # plt.legend("green - start")

        
    
    #     # check_l = np.array(Ypos_list[pairs[0]][smaller_idx_scatter_1:larger_idx_scatter_1])
    #     # def k_smallest_values_with_indices(arr, k):
    #     #     indexed_arr = [(value, index) for index, value in enumerate(arr)]
    #     #     sorted_arr = sorted(indexed_arr, key=lambda x: x[0])
    #     #     return sorted_arr[:k]
    #     # a = k_smallest_values_with_indices(check_l, 10)
    #     # print(a)


    #     buffer_range = 15
    #     bufffer_smaller_idx = get_idx(time_id_sec, smaller_sec  - buffer_range)
    #     bufffer_larger_idx = get_idx(time_id_sec, larger_sec + buffer_range)

    #     # plt.subplot(2,2,3)
    #     ax2= plt.subplot(spec[2:3])
    #     ax3= plt.subplot(spec[4:5])
    #     leading = ax3.plot(xids[bufffer_smaller_idx: bufffer_larger_idx], xvals[bufffer_smaller_idx: bufffer_larger_idx], color = "red", label = "leading")
    #     following = ax3.plot(yids[bufffer_smaller_idx: bufffer_larger_idx], yvals[bufffer_smaller_idx:bufffer_larger_idx],color = "blue", label = "following")
    #     ax3.set_xlabel('Timestamp(min)', fontsize=6) 
    #     ax3.set_title("corresponding convolution plot",fontsize=8)
    #     ax3.legend(loc='upper right', fontsize='x-small')
    #     #ax2.legend((leading, following),(label1, label2))
    #     # ax2.legend("blue--following")
  
    #     #plt.subplot(2,2,4)


    #     smaller_idx_spike_1 = get_idx((normalize_timestamp_second(loc_times_list[pairs[0]], reference)), smaller_sec - buffer_range)
    #     larger_idx_spike_1 = get_idx((normalize_timestamp_second(loc_times_list[pairs[0]], reference)), larger_sec + buffer_range)
    #     smaller_idx_spike_2 = get_idx((normalize_timestamp_second(loc_times_list[pairs[1]], reference)), smaller_sec - buffer_range)
    #     larger_idx_spike_2 = get_idx((normalize_timestamp_second(loc_times_list[pairs[1]], reference)), larger_sec + buffer_range)


    #     # print(type(normalize_timestamp(loc_times_list[pairs[0]], reference)))
    #     # print(len(normalize_timestamp(loc_times_list[pairs[0]], reference)[smaller_idx_spike: larger_idx_spike]))
    #     # print(len(loc_times_list[pairs[0]][smaller_idx_spike: larger_idx_spike]))
    #     # print(loc_times_list[pairs[0]])
    #     # print(MaxHeight_list[pairs[0]])

    #     print(xids[bufffer_smaller_idx])

    #     # ax3= fig.add_subplot(spec[2:])
    #     ax2.vlines(normalize_timestamp(loc_times_list[pairs[0]], reference)[smaller_idx_spike_1: larger_idx_spike_1], ymin=0, ymax=MaxHeight_list[pairs[0]][smaller_idx_spike_1: larger_idx_spike_1], color='red', label = "leading")
    #     ax2.vlines(normalize_timestamp(loc_times_list[pairs[1]], reference)[smaller_idx_spike_2: larger_idx_spike_2], ymin=0, ymax=MaxHeight_list[pairs[1]][smaller_idx_spike_2: larger_idx_spike_2], color='blue', label = "following")
    #     ax2.set_xlabel('Timestamp(min)',fontsize=6) 
    #     ax2.set_ylabel('MaxHeight',fontsize=6)
    #     # ax2.set_xlim(xids[bufffer_smaller_idx], xids[bufffer_larger_idx])
    #     ax2.set_ylim(0, max_yvalues)
    #     ax2.set_title("corresponding spikes(leading)",fontsize=8)
    #     ax2.legend(loc='upper right', fontsize='x-small')
  

    #     plt.show()









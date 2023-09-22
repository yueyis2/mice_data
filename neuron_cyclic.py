import pandas as pd
import matplotlib.pyplot as plt
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


df_list = []
df_new_length = []
tag_list = []
Xpos_list = []
Ypos_list = []
loc_times_list = []
MaxHeight_list = []
MaxWidth_list = []

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

######################
######################

for j in R859_D2:
    # print(j)
    for i in R859_D2[j]:
        # print(i)
        df = pd.read_csv("/Users/evashenyueyi/Downloads/city_data/R859_D2/TT"+ j + "/cl-maze1."+str(i)+".csv")
        maze_name = "R859_D2/TT"+ j + "/cl-maze1."+str(i)
        tag_list.append(maze_name)
        select_cols = df.columns[-5:]
        df.drop(df.index[0:4], inplace=True)
        df_new = df[select_cols].astype(int)

        Xpos = df_new['XPos'].tolist()
        Ypos = df_new['YPos'].tolist()
        loc_times = df_new['Timestamp'].tolist()
        MaxHeight = df_new['MaxHeight'].tolist()
        MaxWidth = df_new['MaxWidth'].tolist()

        # loc_times = [df_new['Timestamp'].tolist()[0]] + df_new['Timestamp'].tolist()

        df_new['Firingtime'] = range(1,len(df)+1)
        data = [make_tuple(df_new['Timestamp'].tolist(), df_new['Firingtime'].tolist())]
        Xpos_list.append(Xpos)
        Ypos_list.append(Ypos)
        loc_times_list.append(loc_times)
        df_new_length.append(df.shape[0])
        MaxHeight_list.append(MaxHeight)
        MaxWidth_list.append(MaxWidth)

    
        spike_len1 = len(MaxHeight)
        t_continuous1 = np.linspace(0, 20, spike_len1)
        norm_conv1 = continuous_function(t_continuous1)/np.max(continuous_function(t_continuous1))
        spike_convo, time_spike = create_spike_convo(loc_times,MaxHeight)
        time_spike = [time_spike[0]]+ list(time_spike)
        convolution_result1 = signal.convolve(spike_convo, norm_conv1, mode='same') ## adding 0 to the unspike points
        convolution_result1 = [0] + list(convolution_result1)


        spike = [make_tuple(time_spike, convolution_result1)]
        df_list.append(spike)


ret, rate_list, xids, yids = fh.make_lead_matrix(fh.flatten(df_list), intfunc=xdy_ydx)
LM, phases, perm, sortedLM, evals = ret

index_list = []
LM_copy = copy.deepcopy(LM)
LM_copy_flatten = LM_copy.flatten()
top_indices = np.argsort(LM_copy_flatten)[-5:]
row_indices, col_indices = np.unravel_index(top_indices, LM_copy.shape)
print("Indices of the 5 largest numbers:")
for i in range(5):
    index_list.append((row_indices[i], col_indices[i]))
    print(f"Number: {LM_copy[row_indices[i], col_indices[i]]}, Index: ({row_indices[i]}, {col_indices[i]})")

print("=======================")
print(index_list)
m = np.argmax(LM_copy_flatten)
r, c = divmod(m, LM.shape[1])
print(r, c)

print("=======================")

# Function defined in plot_helpers.py
fig = plt.figure(figsize=(10,6))
idxs = ph.plot_initial(ret, fig=fig, labels=tag_list)
fig

plt.show()




for pairs in index_list:
    # commented
    # print(pairs)
    # print(type(pairs))

    xtuples, ytuples = get_xy_tuples(pairs[0], pairs[1])
    (xids, xvals), (yids, yvals) = fh.prune(xtuples, ytuples, ids=True)

    xvals = tuple(map(float, xvals))
    yvals = tuple(map(float, yvals))

    xids = normalize_timestamp(xids)
    yids = normalize_timestamp(yids)


    # subplot1
    plt.figure(figsize=(16,8))
    plt.subplot(2,4,1)
    
    
    max_yvalues = max(max(MaxHeight_list[pairs[0]]),max(MaxHeight_list[pairs[1]]))

    plt.vlines(normalize_timestamp(loc_times_list[pairs[0]]), ymin=0, ymax=MaxHeight_list[pairs[0]], color='red', linestyle='dashed')
    plt.xlabel('Timestamps(min)') 
    plt.ylabel('MaxHeight') 
    plt.ylim(0, max_yvalues)
    plt.title('Timeseries Plot for' + str(pairs[0]))

    # plt.show()

    # subplot2
    plt.subplot(2,4,5)

    plt.vlines(normalize_timestamp(loc_times_list[pairs[1]]), ymin=0, ymax=MaxHeight_list[pairs[1]], color='blue', linestyle='dashed')
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
    time_ids, time_axis_end = get_time_axis_min(xids, yids)

    # subplot3
    plt.subplot(2,4,4)
    plt.plot(time_ids,areavals)
    plt.xlabel('Timestamps') 
    plt.ylabel('CumlArea') 
    plt.title('Cumulated Area Plot')
    #plt.xlim(0.0, time_axis_end)
    # plt.show()

    # comment
    # debugging areavals&LM
    print(areavals[-1])
    print(LM[pairs[0]][pairs[1]])

    # subplot4
    # subplot5
    location_plot(Xpos_list[pairs[0]], Ypos_list[pairs[0]], normalize_timestamp(loc_times_list[pairs[0]]), 1)
    location_plot(Xpos_list[pairs[1]], Ypos_list[pairs[1]], normalize_timestamp(loc_times_list[pairs[1]]), 0)


    plt.show()

    
   

    
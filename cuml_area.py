import cyclic_helper as ch
import itertools
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


def cumul_area_noncausal(x, y):
    assert len(x) == len(y)
    z = [0] * len(x)
    z[0] = x[0] * (y[-1] - y[0]) - y[0] * (x[-1] - x[0])
    for n in range(1, len(z)):
        z[n] = z[n-1] - x[n] * y[n-1] + y[n] * x[n-1]
    return [val/2 for val in z]

def get_xy_tuples(xindex, yindex):
    xtuples = cm.df_list[xindex][0]
    ytuples = cm.df_list[yindex][0]
    return xtuples, ytuples


from datetime import datetime, timedelta

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

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

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
    # plt.show()


import func_helpers as fh
import create_matrix as cm
import numpy as np
import matplotlib.pyplot as plt


for pairs in cm.index_list:
    print(pairs)
    print(type(pairs))

    xtuples, ytuples = get_xy_tuples(pairs[0], pairs[1])
    (xids, xvals), (yids, yvals) = fh.prune(xtuples, ytuples, ids=True)
    # print(type(xids))
    # print(xids)

    xvals = tuple(map(float, xvals))
    yvals = tuple(map(float, yvals))
    # print(xids)
    # print(yids)
    # x, y = ch.match_ends(ch.tv_norm(np.nan_to_num(np.asarray(tuple((xvals, yvals))))))

    xids = normalize_timestamp(xids)
    yids = normalize_timestamp(yids)
    # print(xids)
    # print(yids)
    
    def get_time_axis_min(xids, yids):
        time_axis_end = min(xids[-1],yids[-1])
        if len(xids) > len(yids):
            return xids, time_axis_end
        return yids, time_axis_end
    # time_ids, time_axis_end = get_time_axis_min(xids, yids)

    # subplot1
    plt.figure(figsize=(18,10))
    plt.subplot(2,4,1)
    
    # print(len(cm.MaxHeight_list[pairs[0]]))
    # print(cm.MaxHeight_list[pairs[0]])
    # print(len(cm.loc_times_list[pairs[0]]))
    
    max_yvalues = max(max(cm.MaxHeight_list[pairs[0]]),max(cm.MaxHeight_list[pairs[1]]))


    plt.vlines(normalize_timestamp(cm.loc_times_list[pairs[0]]), ymin=0, ymax=cm.MaxHeight_list[pairs[0]], color='red', linestyle='dashed')

    # plt.scatter(list(xids), list(xvals), color='red')
    plt.xlabel('Timestamps(min)') 
    plt.ylabel('MaxHeight') 
    plt.ylim(0, max_yvalues)
    plt.title('Timeseries Plot for' + str(pairs[0]))

    # plt.show()

    # subplot2
    plt.subplot(2,4,5)

    plt.vlines(normalize_timestamp(cm.loc_times_list[pairs[1]]), ymin=0, ymax=cm.MaxHeight_list[pairs[1]], color='blue', linestyle='dashed')

    # plt.scatter(list(yids), list(yvals), color='green')
    plt.xlabel('Timestamps(min)') 
    plt.ylabel('MaxHeight') 
    plt.ylim(0, max_yvalues)
    plt.title('Timeseries Plot for' + str(pairs[1]))
    # plt.show()


    ##for the part convolution continous function with spikes
    from scipy import signal

    def create_spike_convo(loc_times_list,MaxHeight_list,XorY):
        # spike = [(loc_times_list, MaxHeight_list)]
        # time_length = max(loc_times_list) - min(loc_times_list) + 5 ##make sure the length is enough for spike
        # spike_list = [0 for i in range(time_length)]

        # convert to seconds
        # plt.figure()
        # plt.plot(loc_times_list, MaxHeight_list)
        # plt.show()
        loc_times_arr = (np.array(loc_times_list) // 1e5).astype(np.int) ### unit in 1/10s
        MaxHeight_arr = np.array(MaxHeight_list)
        min = loc_times_arr[0]
        max = loc_times_arr[-1]
        loc_times_arr = loc_times_arr - min
        res = np.zeros(max-min+1)
        for i in range(len(MaxHeight_arr)):
            res[loc_times_arr[i]] = MaxHeight_arr[i]
        tmp = (np.arange(min*1e5, min*1e5+len(res)*1e5, 1e5)).astype(np.int)
        # print(len(tmp))
        # print(tmp[0], tmp[-1])
        # print(loc_times_list[0], loc_times_list[-1])
        
        # print(np.max(res))
        # if XorY:
        #     plt.subplot(2,4,2)
        # else:
        #     plt.subplot(2,4,6)
        # plt.plot(res)
        return res, tmp



    a = 1.5
    b = 3
    def continuous_function(x):
        return (np.exp(-x/b) * x**(a-1))
    
    spike_len1 = len(cm.MaxHeight_list[pairs[0]])
    t_continuous1 = np.linspace(0, 20, spike_len1)
    norm_conv1 = continuous_function(t_continuous1)/np.max(continuous_function(t_continuous1))
    spike_convo1, timestamp1 = create_spike_convo(cm.loc_times_list[pairs[0]],cm.MaxHeight_list[pairs[0]], 1)
    convolution_result1 = signal.convolve(spike_convo1, norm_conv1, mode='same')

    spike_len2 = len(cm.MaxHeight_list[pairs[1]])
    t_continuous2 = np.linspace(0, 20, spike_len2)
    norm_conv2 = continuous_function(t_continuous2)/np.max(continuous_function(t_continuous2))
    spike_convo2, timestamp2 = create_spike_convo(cm.loc_times_list[pairs[1]],cm.MaxHeight_list[pairs[1]], 0)
    convolution_result2 = signal.convolve(spike_convo2, norm_conv2, mode='same')


    max_col_value = max(max(convolution_result1), max(convolution_result2))

    # Plot the original list, continuous function, and convolution result
    # plt.figure(figsize=(10, 5))
    # plt.figure()
    plt.subplot(2,4,2)
    plt.plot(convolution_result1)
    plt.ylim(0,max_col_value)
    plt.title("convolution plot for" + str(pairs[0]))
    plt.xlabel('Timestamps in seconds') 
    plt.ylabel('MaxHeight*continuous function sampe') ## need to normalized
    # plt.show()

    # plt.figure()
    plt.subplot(2,4,6)
    plt.plot(convolution_result2)
    plt.ylim(0,max_col_value)
    plt.title("convolution plot for" + str(pairs[1]))
    plt.xlabel('Timestamps in seconds') 
    plt.ylabel('MaxHeight*continuous function sampe') ## need to normalized
    # plt.show()


    xtuples = (timestamp1, convolution_result1)
    ytuples = (timestamp2, convolution_result2)
    (xids, xvals), (yids, yvals) = fh.prune(xtuples, ytuples, ids=True)
    # print(type(xids))
    # print(xids)

    xvals = tuple(map(float, xvals))
    yvals = tuple(map(float, yvals))
    ###

    areavals = cumul_area_iter(xvals, yvals)
    time_ids, time_axis_end = get_time_axis_min(xids, yids)
    # print(len(areavals))

    # subplot3
    plt.subplot(2,4,4)
    plt.plot(time_ids,areavals)
    plt.xlabel('Timestamps') 
    plt.ylabel('CumlArea') 
    plt.title('Cumulated Area Plot')
    #plt.xlim(0.0, time_axis_end)
    # plt.show()

    # debugging areavals&LM
    print(areavals)
    print(cm.LM[pairs[0]][pairs[1]])

    # subplot4
    # subplot5
    location_plot(cm.Xpos_list[pairs[0]], cm.Ypos_list[pairs[0]], cm.loc_times_list[pairs[0]], 1)
    location_plot(cm.Xpos_list[pairs[1]], cm.Ypos_list[pairs[1]], cm.loc_times_list[pairs[1]], 0)


    plt.show()

    
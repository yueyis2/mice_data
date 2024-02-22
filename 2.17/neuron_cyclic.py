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
        if lst[i]>=target:
            return i
    return None



######################
######################

for j in R765RF_D5:
    # print(j)
    for i in R765RF_D5[j]:
        # print(i)
        df = pd.read_csv("/Users/evashenyueyi/Downloads/city_data/R765RF_D5/TT"+ j + "/cl-maze1."+str(i)+".csv")
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
    print(f"Number: {LM_copy[row_indices[i], col_indices[i]]}, Index: ({row_indices[i]}, {col_indices[i]}), Corresponding maze : {tag_list[row_indices[i]], tag_list[col_indices[i]]}")
    

print("=======================")
print(index_list)
m = np.argmax(LM_copy_flatten)
r, c = divmod(m, LM.shape[1])
print(r, c)
# print(tag_list[r], tag_list[c])

print("=======================")

# Function defined in plot_helpers.py
fig = plt.figure(figsize=(10,6))
idxs = ph.plot_initial(ret, fig=fig, labels=tag_list)
fig

plt.show()



i = 0
for pairs in index_list:
    # commented
    # print(pairs)
    # print(type(pairs))

    xtuples, ytuples = get_xy_tuples(pairs[0], pairs[1])
    (xids, xvals), (yids, yvals) = fh.prune(xtuples, ytuples, ids=True)

    raw_ids, raw_end = get_time_axis_min(xids, yids)


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

    # print("======")
    # print(time_ids)


####try to write a function to find prpper smaller_minutes and larger_minutes

    def detect_change(x, delta_x, bound, time_range):
        smaller_minutes = x - delta_x
        larger_minutes = x + delta_x
        smaller_idx = get_idx(time_ids, smaller_minutes)
        mid_idx = get_idx(time_ids, x)
        larger_idx = get_idx(time_ids, larger_minutes)
        y_left = areavals[smaller_idx]
        y_mid = areavals[mid_idx]
        y_right = areavals[larger_idx]
        if abs(y_right - y_left) > bound:
            tmp = make_tuple(x - delta_x, x + delta_x)
            # print(tmp)
            time_range.append(tmp) 
        return time_range

    time_range = []
    for mins in range(1, int(max(time_ids))-1):
        time_range = detect_change(mins, 1, 2.5, time_range)

    print("====the time range need to observe==========")
    print(time_range)



    smaller_minutes = 13
    larger_minutes = 18
    ## get correspinding index in areaval of each time_id
    smaller_idx = get_idx(time_ids, smaller_minutes)
    larger_idx = get_idx(time_ids, larger_minutes)
    # subplot3
    plt.subplot(2,4,4)
    plt.plot(time_ids,areavals, color="blue")
    plt.plot(time_ids[smaller_idx: larger_idx],areavals[smaller_idx: larger_idx], color="red")
    
    plt.xlabel('Timestamps') 
    plt.ylabel('CumlArea') 
    plt.title('Cumulated Area Plot')
    #plt.xlim(0.0, time_axis_end)
    # plt.show()

    # # comment
    # # debugging areavals&LM
    # print(areavals[-1])
    # print(LM[pairs[0]][pairs[1]])

    # subplot4
    # subplot5
    location_plot(Xpos_list[pairs[0]], Ypos_list[pairs[0]], normalize_timestamp(loc_times_list[pairs[0]]), 1)
    location_plot(Xpos_list[pairs[1]], Ypos_list[pairs[1]], normalize_timestamp(loc_times_list[pairs[1]]), 0)


    plt.show()

    ### plot the cuml_posi
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.plot(time_ids,areavals, color="blue")
    plt.plot(time_ids[smaller_idx: larger_idx],areavals[smaller_idx: larger_idx], color="red")
    plt.xlabel('Timestamps') 
    plt.ylabel('CumlArea') 
    plt.title('Cumulated Area Plot')

    import pop_s_2 as ps2
    
    ps2.get_position_plot(smaller_minutes, larger_minutes, "/Users/evashenyueyi/Downloads/city_data/R765RF_D5/Pos.p.ascii.csv")

    plt.show()








###########
    # time_range_for_group_pairs1=[[(12,17),(23,28),(32,37)],
    #                         [(15,25),(28,32)],
    #                         [(22,27),(30,35)],
    #                         [(12,17),(20,25),(25,30)],
    #                         [(8,13),(13,18)]]
    # time_range_for_group_pairs2=[[(13,18),(21,26)],
    #                         [(40,43),(43,47)],
    #                         [(30,35),(50,55)],
    #                         [(18,23)],
    #                         [(30,35),(48,18)]]
    # # for time_range_for_each_pairs in time_range_for_group_pairs:
    # time_range_for_each_pairs = [(30,35),(50,55)]
    #     # print(get_raw_timestamp(12, 17, loc_times_list[pairs[0]]))

    # for time_range in time_range_for_each_pairs:
    #     special_change_ids1 = get_raw_timestamp(time_range[0], time_range[1], loc_times_list[pairs[0]])
    #     position_tuple1 = get_value_tolist(time_pos_dict_list[pairs[0]],special_change_ids1)
    #     difference1 = get_adjacent_time_differences(special_change_ids1) 

    #     special_change_ids2 = get_raw_timestamp(time_range[0], time_range[1], loc_times_list[pairs[1]])
    #     position_tuple2 = get_value_tolist(time_pos_dict_list[pairs[1]],special_change_ids2)
    #     difference2 = get_adjacent_time_differences(special_change_ids2) 

    #     width, height = 500, 400  # Adjust as needed
    #     num_frames1 = len(difference1)
    #     durations1 = difference1
    #     # Define lists of x and y coordinates for each frame
    #     x_coordinates = [each[0] for each in position_tuple1]
    #     y_coordinates = [height - each[1] for each in position_tuple1]
    #     # Create an empty list to store the frames
    #     frames1 = []
    #     # Create frames
    #     for frame in range(num_frames1):
    #         # Create a new blank image for each frame
    #         img = Image.new('RGB', (width, height), color='white')
    #         draw = ImageDraw.Draw(img)

    #         # Get the x and y coordinates for the current frame
    #         x = x_coordinates[frame]
    #         y = y_coordinates[frame]

    #         # Draw the dot at the specified (x, y) coordinates
    #         draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='red')

    #         # Draw the path by overlaying previous frames
    #         for i in range(frame):
    #             x_i = x_coordinates[i]
    #             y_i = y_coordinates[i]
    #             draw.ellipse([x_i - 2, y_i - 2, x_i + 2, y_i + 2], fill='red')
            
    #         for i in range(frame - 1):
    #             x1, y1 = x_coordinates[i], y_coordinates[i]
    #             x2, y2 = x_coordinates[i + 1], y_coordinates[i + 1]
    #             draw.line([(x1, y1), (x2, y2)], fill='red')


    #         # Append the frame to the list with the corresponding duration
    #         frames1.append((img, durations1[frame]))
    #     print(len(frames1))
    #     # Save the frames as a GIF
    #     frames1[0][0].save('moving_plot' + str(pairs[0]) +"for time range"+ str(time_range) +'.gif', save_all=True, append_images=[frame[0] for frame in frames1[1:]], duration=[frame[1] for frame in frames1])

    #     width, height = 500, 400  # Adjust as needed
    #     num_frames2 = len(difference2)
    #     durations2 = difference2
    #     # Define lists of x and y coordinates for each frame
    #     x_coordinates = [each[0] for each in position_tuple2]
    #     y_coordinates = [height - each[1] for each in position_tuple2]
    #     # Create an empty list to store the frames
    #     frames2 = []
    #     # Create frames
    #     for frame in range(num_frames2):
    #         # Create a new blank image for each frame
    #         img = Image.new('RGB', (width, height), color='white')
    #         draw = ImageDraw.Draw(img)

    #         # Get the x and y coordinates for the current frame
    #         x = x_coordinates[frame]
    #         y = y_coordinates[frame]

    #         # Draw the dot at the specified (x, y) coordinates
    #         draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='blue')

    #         # Draw the path by overlaying previous frames
    #         for i in range(frame):
    #             x_i = x_coordinates[i]
    #             y_i = y_coordinates[i]
    #             draw.ellipse([x_i - 2, y_i - 2, x_i + 2, y_i + 2], fill='blue')
            
    #         for i in range(frame - 1):
    #             x1, y1 = x_coordinates[i], y_coordinates[i]
    #             x2, y2 = x_coordinates[i + 1], y_coordinates[i + 1]
    #             draw.line([(x1, y1), (x2, y2)], fill='blue')

    #         # Append the frame to the list with the corresponding duration
    #         frames2.append((img, durations2[frame]))
    #     print(len(frames2))
    #     # Save the frames as a GIF
    #     frames2[0][0].save('moving_plot' + str(pairs[1]) +"for time range"+ str(time_range) +'.gif', save_all=True, append_images=[frame[0] for frame in frames2[1:]], duration=[frame[1] for frame in frames2])

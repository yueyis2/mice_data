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
    starttime = datetime.fromtimestamp(starttime)

    for milliseconds in timestamp_tuple:
        seconds = milliseconds / 6e7
        timestamp_datetime = datetime.fromtimestamp(seconds)
        time_difference = timestamp_datetime - starttime
        # print(time_difference)

        time_difference_minutes.append(time_difference.total_seconds())
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
        plt.subplot(2,3,4)
    else:
        plt.subplot(2,3,5)
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
    print(xids)
    print(yids)
    
    def get_time_axis(xids, yids):
        time_axis_end = min(xids[-1],yids[-1])
        if len(xids) > len(yids):
            return xids, time_axis_end
        return yids, time_axis_end
    time_ids, time_axis_end = get_time_axis(xids, yids)

    # subplot1
    plt.figure(figsize=(16,8))
    plt.subplot(2,3,1)
    plt.scatter(list(xids), list(xvals), color='red')
    
    plt.xlabel('Timestamps(min)') 
    plt.ylabel('CumlFiringRate') 
    plt.title('Timeseries Plot for' + str(pairs[0]))

    # plt.show()

    # subplot2
    plt.subplot(2,3,2)
    plt.scatter(list(yids), list(yvals), color='green')
    plt.xlabel('Timestamps(min)') 
    plt.ylabel('CumlFiringRate') 
    plt.title('Timeseries Plot for' + str(pairs[1]))
    # plt.show()


    areavals = cumul_area_iter(xvals, yvals)
    # print(len(areavals))

    # subplot3
    plt.subplot(2,3,3)
    plt.plot(time_ids,areavals)
    plt.xlabel('Timestamps') 
    plt.ylabel('CumlArea') 
    plt.title('Cumulated Area Plot')
    plt.xlim(0.0, time_axis_end)
    # plt.show()

    # debugging areavals&LM
    print(areavals)
    print(cm.LM[pairs[0]][pairs[1]])

    # subplot4
    # subplot5
    location_plot(cm.Xpos_list[pairs[0]], cm.Ypos_list[pairs[0]], cm.loc_times_list[pairs[0]], 1)
    location_plot(cm.Xpos_list[pairs[1]], cm.Ypos_list[pairs[1]], cm.loc_times_list[pairs[1]], 0)


    plt.show()

    
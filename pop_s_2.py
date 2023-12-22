
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# file =  r'/Users/evashenyueyi/Downloads/city_data/R781_D3/Pos.p.ascii'

# framefile = pd.read_csv(file, encoding='gbk', engine='python',sep=',',delimiter=None,skipinitialspace=True)
# framefile.to_csv("/Users/evashenyueyi/Downloads/city_data/R781_D3/Pos.p.ascii.csv",index=False,sep=',')

def get_position_plot(canva, start_time_min, end_time_min, start_reference, mice_data_file):

    # df = pd.read_csv("/Users/evashenyueyi/Downloads/city_data/R765RF_D5/Pos.p.ascii.csv",low_memory=False)
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
    # print(df_test)
    # print(df.shape)
    # print(df_cleaned.shape)

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

    smaller_idx = get_idx(normalize_timestamp_path(timestamp,start_reference), start_time_min)
    larger_idx = get_idx(normalize_timestamp_path(timestamp,start_reference), end_time_min)


    
    x_coordinates = Xpos[smaller_idx:larger_idx]
    y_coordinates = Ypos[smaller_idx:larger_idx]
    time = normalize_timestamp_path(timestamp[smaller_idx:larger_idx],start_reference)

    # print(x_coordinates[1:100])
    # print(y_coordinates[1:100])
    # print(normalize_timestamp_path(timestamp[smaller_idx:larger_idx][1:100],start_reference))

    print("===========")
    # Saving the lists to a text file
    with open('/Users/eeevashen/Desktop/summer_reserach /trajectory.txt', 'w') as file:
        # Writing each list on a separate line
        # file.write("timerange" + str(start_time_min)+ " to " + str(end_time_min))
        file.write("time: " + ', '.join(map(str, time)) + '\n')
        file.write("xpos: " + ', '.join(map(str, x_coordinates)) + '\n')
        file.write("ypos: " + ', '.join(map(str, y_coordinates)) + '\n')
        # file.write("===========================================")

   
    # with open ('/Users/evashenyueyi/Desktop/summer_reserach /trajectory.txt',"w") as file:
    #     trajectory = []
    #     time = []
    #     for i in range(smaller_idx,larger_idx):
    #         trajectory.append((Xpos[i], Ypos[i]))
    #         time.append(timestamp[i])

    #     file.write("[")
    #     for point in trajectory:
    #         file.write("(" + f"{point[0]}, {point[1]}" + "),")
    #     file.write("]")
    #     file.write(f"{time}")
    



    # if x_coordinates is None or y_coordinates is None:
    #    return None



#     def get_raw_timestamp_position(start_min, end_min, xids):
#             raw_timestamp = []
#             # X_coor=[]
#             # Y_coor=[]
#             startid = xids[0]
#             start = (start_min * 6e7) + startid
#             end = (end_min * 6e7) + startid
#             for ids in xids:
#                 if ids > start and ids < end:
#                     raw_timestamp.append(ids)
#                     # X_coor.append(Xpos[df_cleaned['Timestamp_raw'] == ids])
#                     # Y_coor.append(Ypos[df_cleaned['Timestamp_raw'] == ids])
#             return raw_timestamp

#     # timestamp = get_raw_timestamp_position(12, 17, df_cleaned['Timestamp_raw'])
#     timestamp = get_raw_timestamp_position(start_time_min, end_time_min, df_cleaned['Timestamp_raw'])

#     print(timestamp)

#     df_range1 = df_cleaned[df_cleaned['Timestamp_raw'].isin(timestamp)]
#     X_coor = df_range1['Xpos_raw'].tolist()
#     Y_coor = df_range1['Ypos_raw'].tolist()

#     # print(len(timestamp))
#     # print(len(X_coor))
#     # print(len(Y_coor))

#     # print(min(X_coor),max(X_coor))
#     # print(min(X_coor),max(Y_coor))

#     def get_adjacent_time_differences(lst):
#         differences = [0]
#         for i in range(len(lst) - 1):
#             difference = lst[i + 1] - lst[i]
#             differences.append(difference//1e6)
#         return differences


#     difference1 = get_adjacent_time_differences(timestamp) 

#     # width, height = int(max(X_coor)+30), int(max(Y_coor)+5)  # Adjust as needed
#     # num_frames1 = len(difference1)
#     # durations1 = difference1
#     # Define lists of x and y coordinates for each frame
#     x_coordinates = X_coor
#     y_coordinates = Y_coor

    if len(x_coordinates) == 0 or len(y_coordinates) == 0:
        # plt.subplot(1,2,2)
        ax1.xlim(0, 650)
        ax1.ylim(0, 500)
    else:
        #plt.subplot(1,2,2)
        ax1.plot(x_coordinates, y_coordinates, zorder=1)
        ax1.scatter(x_coordinates[0], y_coordinates[0],  color = "green", zorder=2)
        ax1.scatter(x_coordinates[-1], y_coordinates[-1],  color = "purple", zorder=3)
        ax1.xlim(0, 650)
        ax1.ylim(0, 500)
##   plt.show()

#     def get_raw_timestamp_position(start_min, end_min, xids):
#         raw_timestamp = []
#         # X_coor=[]
#         # Y_coor=[]
#         startid = xids[0]
#         start = (start_min * 6e7) + startid
#         end = (end_min * 6e7) + startid
#         for ids in xids:
#             if ids > start and ids < end:
#                 raw_timestamp.append(ids)
#                 # X_coor.append(Xpos[df_cleaned['Timestamp_raw'] == ids])
#                 # Y_coor.append(Ypos[df_cleaned['Timestamp_raw'] == ids])
#         return raw_timestamp

#     test = get_raw_timestamp_position(18, 20, timestamp)
#     print(test)

# get_position_plot(18, 20, "/Users/evashenyueyi/Downloads/city_data/R781_D2/Pos.p.ascii.csv")




import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# file =  r'/Users/evashenyueyi/Downloads/city_data/R781_D3/Pos.p.ascii'

# framefile = pd.read_csv(file, encoding='gbk', engine='python',sep=',',delimiter=None,skipinitialspace=True)
# framefile.to_csv("/Users/evashenyueyi/Downloads/city_data/R781_D3/Pos.p.ascii.csv",index=False,sep=',')

def get_position_plot_sec(canva, start_time_sec, end_time_sec, start_reference, mice_data_file):

    # df = pd.read_csv("/Users/evashenyueyi/Downloads/city_data/R765RF_D5/Pos.p.ascii.csv",low_memory=False)
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
    # print(df_test)
    # print(df.shape)
    # print(df_cleaned.shape)

    timestamp = df_cleaned['Timestamp_raw'].tolist()
    Xpos = df_cleaned['Xpos_raw'].tolist()
    Ypos = df_cleaned['Ypos_raw'].tolist()



    def normalize_timestamp_path_sec(timestamp_tuple, start_reference):
        time_difference_minutes = []
        starttime = start_reference / 1e6
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

    def get_idx(lst, target):
        for i in range(len(lst)):
            if lst[i]>target:
                return i
        return None

    smaller_idx = get_idx(normalize_timestamp_path_sec(timestamp,start_reference), start_time_sec)
    larger_idx = get_idx(normalize_timestamp_path_sec(timestamp,start_reference), end_time_sec)


    
    x_coordinates = Xpos[smaller_idx:larger_idx]
    y_coordinates = Ypos[smaller_idx:larger_idx]
    time = normalize_timestamp_path_sec(timestamp[smaller_idx:larger_idx],start_reference)

    # print(x_coordinates[1:100])
    # print(y_coordinates[1:100])
    # print(normalize_timestamp_path(timestamp[smaller_idx:larger_idx][1:100],start_reference))

    print("===========")
    # Saving the lists to a text file
    with open('/Users/eeevashen/Desktop/summer_reserach /trajectory.txt', 'w') as file:
        # Writing each list on a separate line
        # file.write("timerange" + str(start_time_min)+ " to " + str(end_time_min))
        file.write("time: " + ', '.join(map(str, time)) + '\n')
        file.write("xpos: " + ', '.join(map(str, x_coordinates)) + '\n')
        file.write("ypos: " + ', '.join(map(str, y_coordinates)) + '\n')
        # file.write("===========================================")

   
    # with open ('/Users/evashenyueyi/Desktop/summer_reserach /trajectory.txt',"w") as file:
    #     trajectory = []
    #     time = []
    #     for i in range(smaller_idx,larger_idx):
    #         trajectory.append((Xpos[i], Ypos[i]))
    #         time.append(timestamp[i])

    #     file.write("[")
    #     for point in trajectory:
    #         file.write("(" + f"{point[0]}, {point[1]}" + "),")
    #     file.write("]")
    #     file.write(f"{time}")
    



    # if x_coordinates is None or y_coordinates is None:
    #    return None



#     def get_raw_timestamp_position(start_min, end_min, xids):
#             raw_timestamp = []
#             # X_coor=[]
#             # Y_coor=[]
#             startid = xids[0]
#             start = (start_min * 6e7) + startid
#             end = (end_min * 6e7) + startid
#             for ids in xids:
#                 if ids > start and ids < end:
#                     raw_timestamp.append(ids)
#                     # X_coor.append(Xpos[df_cleaned['Timestamp_raw'] == ids])
#                     # Y_coor.append(Ypos[df_cleaned['Timestamp_raw'] == ids])
#             return raw_timestamp

#     # timestamp = get_raw_timestamp_position(12, 17, df_cleaned['Timestamp_raw'])
#     timestamp = get_raw_timestamp_position(start_time_min, end_time_min, df_cleaned['Timestamp_raw'])

#     print(timestamp)

#     df_range1 = df_cleaned[df_cleaned['Timestamp_raw'].isin(timestamp)]
#     X_coor = df_range1['Xpos_raw'].tolist()
#     Y_coor = df_range1['Ypos_raw'].tolist()

#     # print(len(timestamp))
#     # print(len(X_coor))
#     # print(len(Y_coor))

#     # print(min(X_coor),max(X_coor))
#     # print(min(X_coor),max(Y_coor))

#     def get_adjacent_time_differences(lst):
#         differences = [0]
#         for i in range(len(lst) - 1):
#             difference = lst[i + 1] - lst[i]
#             differences.append(difference//1e6)
#         return differences


#     difference1 = get_adjacent_time_differences(timestamp) 

#     # width, height = int(max(X_coor)+30), int(max(Y_coor)+5)  # Adjust as needed
#     # num_frames1 = len(difference1)
#     # durations1 = difference1
#     # Define lists of x and y coordinates for each frame
#     x_coordinates = X_coor
#     y_coordinates = Y_coor
    ax1 = canva
    if len(x_coordinates) == 0 or len(y_coordinates) == 0:
        # plt.subplot(2,2,2)
        ax1.set_xlim(0, 650)
        ax1.set_ylim(0, 500)
    else:
        # plt.subplot(2,2,2)
        ax1.plot(x_coordinates, y_coordinates, zorder=1)
        ax1.scatter(x_coordinates[0], y_coordinates[0],  color = "green", zorder=2)
        ax1.scatter(x_coordinates[-1], y_coordinates[-1],  color = "purple", zorder=3)
        ax1.set_xlim(0, 650)
        ax1.set_ylim(0, 500)
##   plt.show()

#     def get_raw_timestamp_position(start_min, end_min, xids):
#         raw_timestamp = []
#         # X_coor=[]
#         # Y_coor=[]
#         startid = xids[0]
#         start = (start_min * 6e7) + startid
#         end = (end_min * 6e7) + startid
#         for ids in xids:
#             if ids > start and ids < end:
#                 raw_timestamp.append(ids)
#                 # X_coor.append(Xpos[df_cleaned['Timestamp_raw'] == ids])
#                 # Y_coor.append(Ypos[df_cleaned['Timestamp_raw'] == ids])
#         return raw_timestamp

#     test = get_raw_timestamp_position(18, 20, timestamp)
#     print(test)

# get_position_plot(18, 20, "/Users/evashenyueyi/Downloads/city_data/R781_D2/Pos.p.ascii.csv")


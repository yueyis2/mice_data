import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# file =  r'/Users/evashenyueyi/Downloads/city_data/R781_D2/Pos.p.ascii'

# framefile = pd.read_csv(file, encoding='gbk', engine='python',sep=',',delimiter=None,skipinitialspace=True)
# framefile.to_csv("/Users/evashenyueyi/Downloads/city_data/R781_D2/Pos.p.ascii.csv",index=False,sep=',')

 # df = pd.read_csv("/Users/evashenyueyi/Downloads/city_data/R765RF_D5/Pos.p.ascii.csv",low_memory=False)
df = pd.read_csv('/Users/evashenyueyi/Downloads/city_data/R781_D2/Pos.p.ascii.csv',low_memory=False)
new_columns = ['Timestamp_raw', 'Xpos_raw','Ypos_raw','Head_angle_raw']
df.columns = new_columns
df['Timestamp_raw'] = df.iloc[:,0]
df['Xpos_raw'] = df.iloc[:,1]
df['Ypos_raw'] = df.iloc[:,2]
df['Head_angle_raw'] = df.iloc[:,3]
# print(df)

# ## since we use the pop.s.ascii data, we may not need to delete the (0,0)
# df_cleaned = df[df['Xpos_raw'] != 0.0]
# df_cleaned = df_cleaned[df_cleaned['Ypos_raw'] != 0.0]
# df_cleaned = df_cleaned[df_cleaned['Xpos_raw'] != -99.0]
# df_cleaned = df_cleaned[df_cleaned['Ypos_raw'] != -99.0]
# df_cleaned = df_cleaned[df_cleaned['Xpos_raw'] <= 610.0]

# df_test = df[df['Xpos_raw'] == 0.0]
# print(df_test)
# print(df.shape)
# print(df_cleaned.shape)

# timestamp = df_cleaned['Timestamp_raw'].tolist()
# Xpos = df_cleaned['Xpos_raw'].tolist()
# Ypos = df_cleaned['Ypos_raw'].tolist()

timestamp = df['Timestamp_raw'].tolist()
Xpos = df['Xpos_raw'].tolist()
Ypos = df['Ypos_raw'].tolist()


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

def get_idx(lst, target):
    for i in range(len(lst)):
        if lst[i]>target:
            return i
    return None

# smaller_idx = get_idx(normalize_timestamp(timestamp),  normalize_timestamp(timestamp)[-1])
# larger_idx = get_idx(normalize_timestamp(timestamp), normalize_timestamp(timestamp)[-1])


plt.plot(Xpos, Ypos, zorder=1)
plt.scatter(Xpos[0], Ypos[0],  color = "red", zorder=2)
plt.scatter(Xpos[-1], Ypos[-1],  color = "orange", zorder=3)
plt.xlim(0, 650)
plt.ylim(0, 500)
plt.show()





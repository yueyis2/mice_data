import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# file =  r'/Users/evashenyueyi/Downloads/city_data/R765RF_D5/Pos.p.ascii'

# framefile = pd.read_csv(file, skiprows=8, encoding='gbk', engine='python',sep=',',delimiter=None,skipinitialspace=True)
# framefile.to_csv("/Users/evashenyueyi/Downloads/city_data/R765RF_D5/Pos.p.ascii.csv",index=False,sep=',')


df = pd.read_csv("/Users/evashenyueyi/Downloads/city_data/R765RF_D5/Pos.p.ascii.csv",low_memory=False)
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

df_test = df[df['Xpos_raw'] == 0.0]
print(df_test)
print(df.shape)
print(df_cleaned.shape)

def get_raw_timestamp_position(start_min, end_min, xids):
        raw_timestamp = []
        # X_coor=[]
        # Y_coor=[]
        startid = xids[0]
        start = (start_min * 6e7) + startid
        end = (end_min * 6e7) + startid
        for ids in xids:
            if ids > start and ids < end:
                raw_timestamp.append(ids)
                # X_coor.append(Xpos[df_cleaned['Timestamp_raw'] == ids])
                # Y_coor.append(Ypos[df_cleaned['Timestamp_raw'] == ids])
        return raw_timestamp

timestamp = get_raw_timestamp_position(33,36, df_cleaned['Timestamp_raw'])

df_range1 = df_cleaned[df_cleaned['Timestamp_raw'].isin(timestamp)]
X_coor = df_range1['Xpos_raw'].tolist()
Y_coor = df_range1['Ypos_raw'].tolist()

print(len(timestamp))
print(len(X_coor))
print(len(Y_coor))

print(min(X_coor),max(X_coor))
print(min(X_coor),max(Y_coor))

def get_adjacent_time_differences(lst):
    differences = [0]
    for i in range(len(lst) - 1):
        difference = lst[i + 1] - lst[i]
        differences.append(difference//1e6)
    return differences


difference1 = get_adjacent_time_differences(timestamp) 

width, height = int(max(X_coor)+30), int(max(Y_coor)+5)  # Adjust as needed
num_frames1 = len(difference1)
durations1 = difference1
# Define lists of x and y coordinates for each frame
x_coordinates = X_coor
y_coordinates = Y_coor
# Create an empty list to store the frames
frames1 = []
# Create frames
for frame in range(num_frames1):
    # Create a new blank image for each frame
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Get the x and y coordinates for the current frame
    x = x_coordinates[frame]
    y = y_coordinates[frame]

    # Draw the dot at the specified (x, y) coordinates
    draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='red')

    # Draw the path by overlaying previous frames
    for i in range(frame):
        x_i = x_coordinates[i]
        y_i = y_coordinates[i]
        draw.ellipse([x_i - 2, y_i - 2, x_i + 2, y_i + 2], fill='red')
    
    for i in range(frame - 1):
        x1, y1 = x_coordinates[i], y_coordinates[i]
        x2, y2 = x_coordinates[i + 1], y_coordinates[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill='red')


    # Append the frame to the list with the corresponding duration
    frames1.append((img, durations1[frame]))
print(len(frames1))
# Save the frames as a GIF
frames1[0][0].save('moving_plot.gif', save_all=True, append_images=[frame[0] for frame in frames1[1:]], duration=[frame[1] for frame in frames1])


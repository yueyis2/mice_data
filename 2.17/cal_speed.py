import pandas as pd
import pos_convo_helper as pch
import math
import matplotlib.pyplot as plt


file_path = "/Users/eeevashen/Desktop/summer_reserach /trajectory_0.5-1.txt"
with open(file_path, 'r') as file:
    for line in file:
        # Splitting the line into label and values
        label, values = line.split(': ')
        values = values.strip().split(', ')

        # Assigning values to the corresponding lists
        if label == 'time':
            read_time = list(map(float, values))
        elif label == 'xpos':
            read_xpos = list(map(float, values))
        elif label == 'ypos':
            read_ypos = list(map(float, values))

df = pd.DataFrame({'timestamp': read_time, 'Xpos': read_xpos, 'Ypos': read_ypos})
df['XYpos'] = df.apply(lambda row: (row['Xpos'], row['Ypos']), axis=1)
first_timestamp = df['timestamp'].iloc[0] * 60
df['timestamp_sec'] = df.apply(lambda row: (row['timestamp']* 60-first_timestamp), axis=1)


def appro_distance(point1, point2):
    square_sum = (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2
    dist = math.sqrt(square_sum)
    return dist

def time_difference(time1, time2):
    diff = time2 - time1
    return diff

#this shift can do as the sliding window ->making the speed-time plot more smooth
df['last_time'] = df['timestamp_sec'].shift(15)
df['last_posi'] = df['XYpos'].shift(15)
df = df[15:]


df['speed'] = df.apply(lambda row: appro_distance(row['XYpos'], row['last_posi'])/time_difference(row['last_time'], row['timestamp_sec']), axis=1)
print(df)

speed = df['speed'].tolist()
time_sec = df['timestamp_sec'].tolist()

average_speed = sum(speed) / len(speed)

# Plotting the two lists
plt.plot(time_sec, speed)

# Adding labels and title
plt.xlabel('time in second')
plt.ylabel('speed(unit as the same as the maze)')
plt.annotate(f'Average Speed: {average_speed:.2f}', xy=(0.1, 0.95), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top')


# Display the plot
plt.show()



# get_speed(file_path)
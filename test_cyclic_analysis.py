import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# TT4_name = [1,2,3,4,5,6,7,8,9,10,11,12]

# for i in TT4_name:
df = pd.read_csv("cl-maze1."+str(6)+".csv")
select_cols = df.columns[-3:]
df.drop(df.index[0:4], inplace=True)
# print(df[select_cols])
new_df = df[select_cols]

def make_tuple(a, b):
    # make any object a and b to a tuple (a,b)
    return (a, b)

new_df['Space'] = new_df.apply(lambda row: make_tuple(row['XPos'], row['YPos']), axis=1)
print(new_df)

# show firing location as scatter plot
plt.scatter(new_df['XPos'], new_df['YPos'])
plt.title('Firing location')
plt.xlabel('XPos')
plt.ylabel('YPos')
plt.show()

# show firing time as line plot
plt.plot(new_df['Timestamp'])
plt.title('Line Plot')
plt.xlabel('Time')
plt.show()


# run cyclic analysis:

# data = [[(intify(f.timestamps),f.firingrates.tolist()) for f in \
#         n.fields.values()] for n in global_neurons.values()]

# ret = make_lead_matrix(flatten(data), intfunc=xdy_ydx)
# LM, phases, perm, sortedLM, evals = ret


# ts_names = list()
# ts_names = ["--".join((nn, f)) for nn, gn in global_neurons.items() for f in \
#             gn.fields.keys()]

# # Function defined in plot_helpers.py
# fig = plt.figure(figsize=(25,12))
# idxs = plot_initial(ret, fig=fig)
# fig
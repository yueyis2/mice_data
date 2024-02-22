import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

def location_plot(Xpos, Ypos, loc_times):
# Example location data with corresponding times
    latitudes = Ypos
    longitudes = Xpos
    times = loc_times  # Example time values (can be in hours)

    # Create a colormap
    cmap = ListedColormap(['blue', 'green', 'yellow', 'red'])
    normalize = Normalize(vmin=min(times), vmax=max(times))

    # Create a scatter plot of locations with colors based on time
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes, latitudes, c=times, cmap=cmap, marker='o', s=10, norm=normalize)

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Time')


    # Set labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Location Plot with Colored Points Based on Time')

    # Show the plot
    plt.grid(True)
    plt.show()

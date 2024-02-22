import requests
import numpy as np
from heapq import merge
from functools import partial
from itertools import combinations, tee
from cyclic_helper import sort_lead_matrix


def pairwise(iterable):
    """Iterate over an iterable two elements at a time."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def gaussian(x, mu, b=0, k=1):
    """Evaluate a unit variance gaussian with mean k*mu with noise b*rand().

    Parameters
    ----------
    x
        Plotting or evaluation variable
    mu
        Fundamental mean value
    b
        Spread of random noise around the mean value
    k
        Higher offsets of mean value
    """
    randval = np.random.rand()
    return (mu, b, k, randval), np.exp(-(x-mu*k+b*randval)**2/2)


def sensed_gaussian(x, params, dist):
    """Evaluate a unit variance gaussian with mean k*mu with noise b*rand().

    Parameters
    ----------
    params
        A tuple consisting of mu, b, k from the gaussian method that generated
        the pulse
    dist
        Distance of the sensor node from source node
    """
    mu, b, k, randval = params
    return np.exp(-(x-dist-mu*k+b*randval)**2/2)


def find_nearest(array, value):
    """Find the element in 1-D array that is closest to value.

    Parameters
    ----------
    array
        A 1-D numpy array
    value
        The value to look for in array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def flatten(regular_list):
    """Flattens a list using list comprehensions.

    Parameters
    ----------
    regular_list:
        A singly nested list to flatten.
    """
    return [item for sublist in regular_list for item in sublist]


def download_file_from_google_drive(id, destination):
    """Download a drive from Google Drive give id from shareable link.

    Parameters
    ----------
    id:
        The file identifier from a shareable link
    destination:
        The filename to save as on local disk
    """
    URL = "https://docs.google.com/uc?export=download&confirm=9iBg"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    """ Function to filter out some Cookie business from Google and
        extract the actual data

    Parameters
    ----------
    response:
        The return value from a requests GET request
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    """ Function to open write the proper response content from a
        requests GET response to local disk.

    Parameters
    ----------
    response:
        The filtered return value from a requests GET request
    destination:
        A filename or file object denoting where to save file on
        local disk
    """
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def intify(arr):
    """Convert array into list of ints excepting nans in array.

    Parameters
    ----------
    arr:
        array to intify
    """
    return list(map(lambda x: x if np.isnan(x) else int(x), arr))


def unique_justseen(iterable, key=None):
    "List unique elements, preserving order. Remember only the element just seen."
    from operator import itemgetter
    from itertools import groupby

    return map(next, map(itemgetter(1), groupby(iterable, key)))


def prune(x, y, ids=True):
    """Given two (possibly repeating) time stamped sequences 'prune' them.
    Parameters
    ----------
    x
        A tuple of (timestamps, data)
    y
        A tuple of (timestamps, data)
    """
    # Tag them so we know who-is-who after merge
    def tag(label, arr):
        # print(arr[0])
        # print(arr)
        # print(type(arr))
        labels = [label for _ in range(len(arr[0]))]
        return zip(*arr, labels)
    

    
    *tagged, = map(list, [tag(l, arr) for l, arr in zip([1, -1], [x, y])])
    


    # Merge/sort-merge them 
    merged = merge(*tagged, key=lambda x:x[0])
    *pruned, = unique_justseen(merged, key=lambda x:x[2])
    seperated = [list(filter(lambda x:x[2]==z, pruned)) for z in [1, -1]]

    (xids, xvals, _), (yids, yvals, _)  = [zip(*z) for z in seperated]
    N = min(map(len, [xvals, yvals]))

    if ids:
        # print("==================")
        # print((xids[:N], xvals[:N]), (yids[:N], yvals[:N]))
        return (xids[:N], xvals[:N]), (yids[:N], yvals[:N])
    else:
        return xvals[:N], yvals[:N]


def make_pairs(data):
    """Make pruned pairs from a list of data."""
    # (xids, xvals), (yids, yvals) = prune(xx, yy, ids=True)
    # return [prune(*pair) for pair in combinations(data, 2)]

    # toreturn_list = []
    # for pair in combinations(data, 2):
    #     toreturn_list.append(prune(*pair))
    # return toreturn_list

    rate_list = []
    xids_list = []
    yids_list = []
    for pair in combinations(data, 2):
        (xids, xvals), (yids, yvals) = prune(*pair) 
        rate_list.append((xvals, yvals))
        xids_list.append(xids)
        yids_list.append(yids)
    return rate_list, xids_list, yids_list

import cyclic_helper as ch
def xdy_ydx(pair):
    """Calculate the area integral between a pair of timeseries data."""
    # x, y = np.asarray(pair[0]), np.asarray(pair[1])
    x, y = ch.match_ends(ch.tv_norm(ch.mean_center(np.nan_to_num(np.asarray(pair)))))
    return (x.dot(ch.cyc_diff(y)) - y.dot(ch.cyc_diff(x))) / 2

def make_lead_matrix(data_list, intfunc):
    """Manually create the lead matrix from a list & integration function.

    Parameters
    ----------
    data_list
        A list of tuples (timestamps, firingrates)
    intfunc
        A function to create the area value from a pair of time series
    """
    N = len(data_list)
    lead_matrix = np.zeros((N, N))
    upper_triangle = [(i, j) for i in range(N) for j in range(i + 1, N)]

    rate_list, xids, yids = make_pairs(data_list)
    for *index, pair in zip(upper_triangle, rate_list):
        index = index.pop()
        # print("=======================")
        # print(index)
        # print("pair is:" + str(pair))
        area = intfunc(pair)
        # print(area)
        lead_matrix[index] = area
        lead_matrix[tuple(reversed(index))] = - area

    return sort_lead_matrix(lead_matrix, 1), rate_list, xids, yids


def moving_average(n, x):
    """ Calculate the n-point moving averge along x. """

    return np.convolve(x, np.ones(n), 'valid') / n


def get_stats(state, start):
    """Plot data related to an US State.

    Parameters
    ----------
    state
          A state instance with following properties:
              - abbrev : A 2 letter abbreviation (string)
              - raw : A tuple of (dates, data)
              - smooth: A tuple of (dates, data)
              - logts: A tuple of (dates, data)
    start
           The date from which to start collating data
    
    Returns
    -------
    A tuple of dates and data arrays
    """
    items = ['date', 'positive', 'positiveIncrease']
    columns = state[items].set_index('date').sort_index().loc[start:]
    *dates, = map(np.datetime64, columns.index.tolist())
    daily_cases = columns['positiveIncrease'].values
    
    return np.asarray(dates), daily_cases

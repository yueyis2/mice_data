import numpy as np
from numpy import mod, outer, mean, argsort, std
from numpy import pi, linspace, newaxis, roll, zeros, angle
from numpy.linalg import norm, eig
from scipy.signal import detrend


def match_ends(Z):
    """ Adjust data so starting and ending points match in the timeseries. """
    n = Z.shape[1]  # No of columns
    return Z - outer(Z[:, n - 1] - Z[:, 0], linspace(0, 1, n))


def mean_center(Z):
    """ Mean center the data matrix Z. """
    return Z - mean(Z, axis=1)[:, newaxis]


def total_variation(Z, axis=0):
    """ Normalize the data by the total variation (abs). """
    dZ = Z - roll(Z, 1, axis=axis)
    return norm(dZ, axis=axis).squeeze()


def quad_norm(Z):
    """Normalize vector(s) Z so that the quadratic sum equals to 1."""
    norms = norm(Z, axis=1)[:, newaxis]
    normed_z = np.divide(Z, norms, out=np.zeros_like(Z), where=norms!=0)
    return normed_z


def tv_norm(Z):
    """ Normalize vector(s) Z so that the quadratic variation is 1. """
    norms = total_variation(Z, axis=1)[:, newaxis]
    normed_z = np.divide(Z, norms, out=np.zeros_like(Z), where=norms!=0)
    return normed_z


def std_norm(Z):
    """ Normalize vector(s) Z so that standard deviation is 1. """
    norms = std(Z, axis=1)[:, newaxis]
    normed_z = np.divide(Z, norms, out=np.zeros_like(Z), where=norms!=0)
    return normed_z

def remove_linear(Z):
    """ Remove linear trends in the data. """
    return detrend(Z)


def cyc_diff(x):
    """ Do cyclic differentiation. """
    return np.diff(np.concatenate(([x[-1]], x)))


def create_lead_matrix(data):
    """ Create the lead matrix from the data. """
    N, time_steps = data.shape
    lead_matrix = zeros((N, N))

    # Create index list of upper_triangle (lower part is anti-symmetric)
    # This is good for small values of N.
    # For larger values, just do a double "for" loop
    upper_triangle = [(i, j) for i in range(N) for j in range(i + 1, N)]

    for (i, j) in upper_triangle:
        x, y = data[i], data[j]
        d = x.dot(cyc_diff(y)) - y.dot(cyc_diff(x))
        lead_matrix[i, j] = d
        lead_matrix[j, i] = -d

    return lead_matrix


def area_val(x, y):
    """ Return the area integral between two arrays x and y. """
    return x.dot(cyc_diff(y)) - y.dot(cyc_diff(x)) 


def sort_lead_matrix(LM, p=1):
    """" Sort the lead matrix using the phases of the p-th eigenvector.

    Parameters
    ----------
    LM
        The Lead matrix
    p
        The eigenvector index to use (integer: 0, 1, ...)
    """
    # The first input should be the matrix to be sorted, the second is the
    # phase or eigenvector to use (default 1).
    evals, phases = eig(LM)
    phases = phases[:, 2 * p - 2]
    sorted_ang = np.sort(mod(angle(phases), 2 * pi))
    dang = np.diff(np.hstack((sorted_ang, sorted_ang[0] + 2 * pi)))
    shift = np.argmax(np.abs(dang))
    shift = (shift + 1) % phases.size

    shift = sorted_ang[shift]
    perm = argsort(mod(mod(angle(phases), 2 * pi) - shift, 2 * pi))
    sortedLM = LM[perm].T[perm].T

    # print("!!!---------!!!!")
    # print(phases)
    # print(perm)
    return LM, phases, perm, sortedLM, evals

def cyclic_analysis(data, p):
    """ Wrapper function to perform cyclicity analysis. 

    Parameters
    ----------
    data
        Appropriately normalized data matrix 
    p
        Eigenvector index/cycle to consider
    """
    lead_matrix = create_lead_matrix(match_ends(data))
    return sort_lead_matrix(lead_matrix, p) 


norms = {None: ('Leave Intact', lambda t: t),
         'sqr': ('Unit Squares', quad_norm),
         'tv': ('Unit Quadratic Variation', tv_norm), 
         'std': ('Unit Standard Deviation', std_norm)}

trend_removals = {None: ('None', lambda t: t),
                  'linear': ('Remove Linear Trend', remove_linear)}
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

tfunctions = {'rms': lambda x:np.sqrt(np.mean(np.square(np.abs(x)))),
                      'mean': lambda x:np.mean(np.abs(x))}


def plot_evec(fig, ax, vec, tfunc='rms', out=False):
    """ Helper function to plot elements of eigenvector around complex plane.
    
    Parameters
    ----------
    fig
        The matplotlib Figure object to plot on
    ax
        The matplotlib.axes object to use 
    vec
        The eigenvector to plot
    tfunc
        The function to put a threshold around the origin
    out
        Boolean specifying whether to highlight points outside the limit
        obtained via usage of tfunc
    """

    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    
    lim = max(np.abs(np.amax(vec)), np.abs(np.amin(vec)))
    circ = plt.Circle((0, 0), lim, color='k', fill=False)
    x = np.real(vec).flatten()
    y = np.imag(vec).flatten()
    
    ax.plot(x, y, 'ko:', zorder=1)
    ax.plot(x[0], y[0], 'ro', zorder=2)
    ax.add_artist(circ)
    ax.grid()
    
    plt.xlim(right=1.1 * lim, left=-1.1 * lim)
    plt.ylim(top=1.1 * lim, bottom=-1.1 * lim)
    lim = tfunctions[tfunc](vec)
    circ = plt.Circle((0, 0), lim, color='b', fill=False, linestyle='dotted')
    ax.add_artist(circ)
    
    if out:
        idx = np.where(np.abs(vec)>lim)
        ax.scatter(x[idx], y[idx], c='blue', zorder=3)
        return ax, idx

    return ax


def highlight_cell(x,y, ax=None, **kwargs):
    """ Highlight the cell indexed by x,y in grid held in ax.

    Parameters
    ----------
    x
        Row index of the cell in a grid to highlight
    y
        Column index of the cell in a grid to highlight
    ax
        A matplotlib axes object containing the plotted grid
    """

    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def heatmap(data, row_labels, col_labels, ax=None, rot=-30,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize = 4)
    ax.set_yticklabels(row_labels, fontsize = 6)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=rot, ha="left", va="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # for edge, spine in ax.spines.items():
    #    spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im 


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    from matplotlib import ticker

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
  
  
def multipage(filename, figs=None, dpi=200):
  """ Writes open figures to a PDF file 

  Parameters
  ----------
  filename:
      A string or file object
  figs:
      A list of open figures, default is None
  dpi:
      The dots per inch (quality) of the output PDF
  """
  
  import matplotlib.pyplot as plt

  pp = PdfPages(filename)
  if figs is None:
      figs = [plt.figure(n) for n in plt.get_fignums()]
  for fig in figs:
      fig.savefig(pp, format='pdf')
  pp.close()


def plot_initial(ret, grid=None, fig=None, labels=False):
  """Plot output from a cyclicty analysis.

  Parameters
  ----------
  ret
        A tuple returned by the `sort_lead_matrix` function in `cyclic_analysis.py`
  
  Returns
  -------
  idxs
        The position of elements of the leading eigenvector that are outside
        the root mean square value are returned as indices
  """

  LM, phases, perm, sortedLM, evals = ret

  oldfontsize = plt.rcParams['font.size']
  plt.rcParams['font.size']=14

  if grid is None:
#    fig = plt.figure(figsize=(20,10), dpi=200, constrained_layout=True)
    left, right = fig.add_gridspec(1,2)
    left_gs = left.subgridspec(2,2)
    right_gs = right.subgridspec(1,1)

    stuff = [fig.add_subplot(left_gs[a,b]) for a in range(2) for b in range(2)]
    r1c1, r1c2, r2c1, r2c2 = stuff
    rax = fig.add_subplot(right_gs[0,0])
  elif fig is None:
    raise NotImplementedError
  else:
    subgrid = grid.subgridspec(2, 4)
    stuff = [fig.add_subplot(subgrid[a,b]) for a in range(2) for b in range(2)]
    r1c1, r1c2, r2c1, r2c2 = stuff
    rax = fig.add_subplot(subgrid[:, 2:])


  r1c1.stem(np.abs(evals), use_line_collection=True)
  r1c1.set_xlabel('Eigenvalues')
  r1c1.set_ylabel('Absolute value of eiegenvalues')
#  r1c1.set_title('Eigenvalue drop off')

  r1c2.stem(np.cumsum(np.abs(evals)[::2])/np.abs(evals)[::2].sum(), 
          use_line_collection=True)
  r1c2.set_xlabel('Every second eigenvalue')
  r1c2.set_ylabel('Cumulative contribution of eigenvalues')
#  r1c2.set_title('Eigenvalue contribution to totals')

  if labels:
      plabels = np.asarray(labels)[perm]
      heatmap(LM, labels, labels, rot=90, ax=r2c1)
      heatmap(sortedLM, plabels, plabels, rot=90, ax=r2c2)
  else:
      r2c1.imshow(LM)
      r2c2.imshow(sortedLM)

 # r2c1.set_title('Obtained lead matrix')
 # r2c2.set_title('Lead matrix after sorting')

  ax, idxs = plot_evec(fig, rax, phases, 'rms', out=True)
  ax.set_title('Leading eigenvector components & RMS value')
  
  plt.rcParams['font.size'] = oldfontsize
  
  return idxs


def plot_state(state, ax, item=None, yadjust=False):
  """Plot data related to an US State.

  Parameters
  ----------
  state
        A state instance with following properties:
            - abbrev : A 2 letter abbreviation (string)
            - raw : A tuple of (dates, data)
            - smooth: A tuple of (dates, data)
            - logts: A tuple of (dates, data)
  ax
        The pyplot axis to plot on
  item 
        One of 'raw', 'smooth' or 'logts'
  """

  item, both = ('raw', True) if item is None else (item, False)
  mapper = {None: 'bar', 'raw': 'bar', 'smooth':'plot', 'logts':'plot'}
  stuff, plotter = [getattr(*z) for z in zip([state, ax],[item,
                    mapper[item]])]
  cax = plotter(*stuff)
  
  if both:
      ax.plot(*getattr(state, 'smooth'), 'k')
  
  _, ymax = ax.get_ylim()
  if yadjust:
      ax.set_ylim(0, ymax)
  
  ax.xaxis.set_ticks([])
  ax.set_xlabel(repr(state))
  return ax


def colormap(mapname, vals):
  """ Creates some colormap related objects give the name of a colorm
  and an array `vals` of values that need to  be color mapped. """
  
  from matplotlib.colors import Normalize
  import matplotlib.cm as cm
  
  norms = Normalize(vmin=vals.min(), vmax=vals.max(), clip=True)
  cmap = cm.ScalarMappable(norm=norms, cmap=plt.get_cmap(mapname))
  *cvals, = map(cmap.to_rgba, vals)

  return cvals, cmap, norms
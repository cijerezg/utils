import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.colors as colors
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_dir(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def implot_percentile(x, path, name, w=1, h=1, p=98):
    """
    Plot images setting max value and min value to the chosen percentile.

    It saves the .png file to path + name.

    Parameters
    ----------
    x : array
        Shape should be (x,y,z), where x is the number of images, and (y,z) is
        the images.
    w : integer
        The number of subplots along the horizontal axis
    h : integer
        The number of subplots along the vertical axis
    path : string
           The path where the image will be saved
    name : string
           The name with which the file will be saved

    Returns
    -------
    It saves the image in the specified directory
    """
    fig, axes = plt.subplots(h, w, figsize=(26, 18))
    if h != 1 and w != 1:
        axes = axes.flatten()
    for i in range(x.shape[0]):
        vmax = np.percentile(x[i, :, :], p)
        vmin = np.percentile(x[i, :, :], 100 - p)
        if x.shape[0] == 1:
            axes.imshow(x[i, :, :], aspect='auto', vmax=vmax, vmin=vmin)
        else:
            axes[i].imshow(x[i, :, :], aspect='auto', vmax=vmax, vmin=vmin)
    create_dir(path)
    plt.savefig(path + '/' + name, bbox_inches='tight')
    plt.close()


def lineplot_seaborn(data, path, name, classes, labels=None):
    """
    Generate a lineplot; the line that calculates std at every point.

    It saves the .png file to path + name.

    Parameters
    ----------
    data: list
       Each element in the list is a class, and has shape (y,z),
       y is the number of samples per class, and z is data length.
    path: string
          The path where the image will be saved
    name: string
          The name with which the file will be saved
    classes: dict
             The keys should 0,1,..,n containing the class name
    labels: dict
            Labels to the plot. Keys are 'xlabel' and 'ylabel'
    """
    types = len(data)
    sns.set(rc={'figure.figsize': (26, 18)}, font_scale=3)
    x = [
        np.arange(
            data[i].shape[1])[
            np.newaxis,
            :].repeat(
                data[i].shape[0],
            0) for i in range(types)]
    hue = [np.full_like(data[i], classes[i], dtype='object')
           for i in range(types)]
    d_flat = []
    x_flat = []
    hue_flat = []
    for i in range(types):
        d_flat.extend(data[i].flatten())
        x_flat.extend(x[i].flatten())
        hue_flat.extend(hue[i].flatten())
    sns.lineplot(x=x_flat, y=d_flat, hue=hue_flat, ci='sd')
    if labels is not None:
        plt.xlabel(labels['xlabel'])
        plt.ylabel(labels['ylabel'])
    create_dir(path)
    plt.savefig(path + '/' + name, bbox_inches='tight')
    plt.close()


def upper_triangular_image(data, path, name, ver=1, hor=1):
    """
    Plot an upper triangular image.

    The colormap is design to have strong contrast with zero. It saves the
    .png file to path + name.

    Parameters
    ----------
    data: list
      Each element in the list is an image (array) to be plotted.
    path: string
          The path where the image will be saved.
    name: string
          The name with which the file will be saved.
    ver: integer (default 1)
      Number of rows for the subplots.
    hor: integer (default 1)
      Number of columns for the subplots.
    """
    fig, axes = plt.subplots(ver, hor, figsize=(15 + 2, ver / hor * 15 + 2))
    if hor != 1 and ver != 1:
        axes = axes.flatten()
    col_pos = plt.cm.Oranges(np.linspace(0.35, 0.95, 256))
    col_neg = plt.cm.Blues_r(np.linspace(0.05, 0.65, 256))
    col_med1 = plt.cm.Greens_r(np.linspace(0.05, 0.65, 256))
    col_med2 = plt.cm.Purples(np.linspace(0.35, 0.95, 256))
    all_cols = np.vstack((col_neg, col_med1, col_med2, col_pos))
    w_map = colors.LinearSegmentedColormap.from_list('w_map', all_cols)
    divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=np.pi / 2, vmax=np.pi)
    for i in range(len(data)):
        if len(data) == 1:
            arr = np.ma.masked_where(data[i] == 0, data[i])
            im = axes.imshow(arr, cmap=w_map, norm=divnorm, aspect='auto')
            axes.axis('off')
        else:
            arr = np.ma.masked_where(data[i] == 0, data[i])
            im = axes[i].imshow(arr, cmap=w_map, norm=divnorm, aspect='auto')
            axes[i].axis('off')
    if len(data) == 1:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.18)
        fig.colorbar(im, cax=cax, orientation='vertical')
    else:
        cax, kw = mpl.colorbar.make_axes([ax for ax in axes])
        plt.colorbar(im, cax=cax, **kw)
    plt.savefig(path + '/' + name, bbox_inches='tight')
    plt.close()


def regular_plot(data, path, name, ver=1, hor=1):
    """
    Plot a regular plot.

    It saves the .png file to path + name.

    Parameters
    ----------
    data: list
      Each element in the list is an array to be plotted.
    path: string
          The path where the image will be saved.
    name: string
          The name with which the file will be saved.
    ver: integer (default 1)
      Number of rows for the subplots.
    hor: integer (default 1)
      Number of columns for the subplots.
    """
    fig, axes = plt.subplots(ver, hor, figsize=(30 + 2, ver / hor * 15 + 2))
    if hor != 1 and ver != 1:
        axes = axes.flatten()
    for i in range(len(data)):
        if len(data) == 1:
            axes.plot(data[i])
            axes.set_xlabel('Iteration')
            axes.set_ylabel('Norm')
            axes.grid(True)
        else:
            axes[i].plot(data[i])
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylabel('Norm')
            axes[i].grid(True)
    plt.savefig(path + '/' + name, bbox_inches='tight')
    plt.close()


def multi_axes_plot(data, path, name, titles):
    """
    Plot a single regular plot with n number of axes.

    It saves the .png file to path + name.

    Parameters
    ----------
    data: list
      Each element in the list is an array to be plotted.
    path: string
          The path where the image will be saved.
    name: string
          The name with which the file will be saved.
    ver: integer (default 1)
      Number of rows for the subplots.
    hor: integer (default 1)
      Number of columns for the subplots.
    """
    fig, host = plt.subplots(1, 1, figsize=(30, 15))
    pars = []
    pars.append(host)
    length = len(data)
    host.grid(True)
    for i in range(length-1):
        par = host.twinx()
        pars.append(par)
    host.set_xlim(0, len(data[0]))
    host.set_xlabel('Iterations', fontsize=22)
    colors = np.linspace(0, 1, len(data))
    lns = []
    for index, axis in enumerate(pars):
        axis.set_ylabel(titles[index], fontsize=22)
        axis.set_ylim(np.min(data[index])-.01, np.max(data[index])+0.01)
        color = plt.cm.Dark2(colors[index])
        p, = axis.plot(data[index], color=color,
                       label=titles[index], linewidth=2.5)
        axis.tick_params(labelsize=20)
        lns.append(p)
    host.legend(handles=lns, loc='best', fontsize=22)
    if length > 2:
        for i in range(2, length):
            pars[i].spines['right'].set_position(('outward', 80*(i-1)))
    for i, axis in enumerate(pars):
        axis.yaxis.label.set_color(lns[i].get_color())
    plt.savefig(path + '/' + name, bbox_inches='tight')
    plt.close()

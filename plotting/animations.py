"""File for animations."""

from netgraph import Graph
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import numpy as np
import networkx as nx
import pdb


def anim_3d_data(data, size_x, size_y, ver, hor, name, path, shades=True):
    """
    Create 3d animation that rotates.

    Parameters
    ----------
    data : list
           Each element contains an array with size (i,n,3), where i is the
           number of frames, n is the length of the data, and the three
           dimensions: x, y, z. It must be that n=size_x*size_y
    size_x : integer
             Number of points along the x axis
    size_y : integer
             Number of points along the y axis
    ver : integer
          Vertical subplots
    hor : integer
          Horizontal subplots
    name : string
           The name with wich the file will be saved
    path : string
           Path where video will be saved
    shades : bool (True by default)
             Helps reduce artifacts if False

    Returns
    -------
    It saves a video in the speficied directory
    """
    fig, axes = plt.subplots(
        ver, hor, figsize=(15 + 2, ver / hor * 15 + 2),
        subplot_kw={'projection': '3d'}, facecolor='black')
    fig.tight_layout()
    axes = axes.flatten()
    lines = []
    xw = np.reshape(data[0][0, :, 0], (size_x, size_y, 1))
    yw = np.reshape(data[0][0, :, 1], (size_x, size_y, 1))
    zw = np.reshape(data[0][0, :, 2], (size_x, size_y, 1))
    ones = np.ones((xw.shape))
    xw = (xw - xw.min()) / (xw.max() - xw.min())
    yw = (yw - yw.min()) / (yw.max() - yw.min())
    zw = (zw - zw.min()) / (zw.max() - zw.min())
    w = np.concatenate((xw, yw, zw, ones), 2)

    def runs(it, data, line):
        print(it)
        for nd, (ax, dat) in enumerate(zip(axes, data)):
            x = np.reshape(dat[it, :, 0], (size_x, size_y))
            y = np.reshape(dat[it, :, 1], (size_x, size_y))
            z = np.reshape(dat[it, :, 2], (size_x, size_y))
            ax.clear()
            ax.plot_surface(x, y, z, facecolors=w, shade=shades)
            ax.set_xlim3d([1.1 * data[nd][:, :, 0].min(),
                           1.1 * data[nd][:, :, 0].max()])
            ax.set_ylim3d([1.1 * data[nd][:, :, 1].min(),
                           1.1 * data[nd][:, :, 1].max()])
            ax.set_zlim3d([1.1 * data[nd][:, :, 2].min(),
                           1.1 * data[nd][:, :, 2].max()])
            ax.set_box_aspect((np.ptp(data[nd][:, :, 0]), np.ptp(
                data[nd][:, :, 1]), np.ptp(data[nd][:, :, 2])))
            angle = (it % (360 * 2)) / 2
            angle = angle - 90
            ax.view_init(angle, 30)
            ax.set_axis_off()
            ax.set_facecolor('black')
    ani = animation.FuncAnimation(
        fig, runs, data[0].shape[0], fargs=(data, lines),
        interval=48, blit=False)
    ani.save(path + '/' + name + '.mp4')
    plt.close()


def graph_anim(data, loc, ver, hor, titles, filename, path):
    """
    Create a neural network animation.

    Parameters
    ----------
    data : list
           Each element
    loc : list
           Node location
    ver : integer
          Vertical subplots
    hor : integer
          Horizontal subplots
    titles : list
          It contains titles for subplots
    filename : string
           The name with wich the file will be saved
    path : string
           Path where video will be saved

    Returns
    -------
    It saves a video in the specified directory
    """
    frames = len(data[0])
    fig, axes = plt.subplots(ver, hor, facecolor='black')
    axes = axes.flatten()
    positions = []
    for j, (dat, p) in enumerate(zip(data, loc)):
        G = nx.from_numpy_matrix(dat[0])
        nodes = list(G.nodes)
        pos = {}
        ind = 0
        for i, node in enumerate(nodes):
            before = p[ind]
            current = p[ind + 1] - before
            pos[node] = (2 * ind, i - before - (current - 1) / 2)
            if i >= p[ind + 1] - 1:
                ind += 1
        nx.draw(G, pos, ax=axes[j])
        positions.append(pos)
    col_pos = plt.cm.Oranges(np.linspace(0.35, 0.95, 256))
    col_neg = plt.cm.Blues_r(np.linspace(0.05, 0.65, 256))
    all_cols = np.vstack((col_neg, col_pos))
    w_map = colors.LinearSegmentedColormap.from_list('w_map', all_cols)
    def update(it, data, pos, ax, w_map, titles):
        node_update = np.log(len(ax))*len(ax)
        if it % 50 == 0:
            print(it)
        for i, (dat, p) in enumerate(zip(data, pos)):
            ax[i].clear()
            ax[i].set_title(titles[i], y=.94)
            G = nx.from_numpy_matrix(dat[it])
            edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
            nx.draw(
                G,
                p,
                node_color='#24FF00',
                edgelist=edges,
                edge_color=weights,
                width=weights,
                edge_cmap=w_map,
                edge_vmin=-5,
                edge_vmax=5,
                ax=ax[i],
                node_size=50/node_update)
    ani = animation.FuncAnimation(fig, update, frames=frames, fargs=(
        data, positions, axes, w_map, titles), interval=80)
    plt.subplots_adjust(0, 0, 1, .97, 0.05, 0.03)
    ani.save(path + '/' + filename + '.mp4', dpi=500)
    plt.close()

    
def im_anim(data, ver, hor, titles, vmin, vmax, filename, path):
    """
    Create animations from series of images.

    Parameters
    ----------
    data : list
           Each element contains a sequence of images.
    ver : integer
          Vertical subplots
    hor : integer
          Horizontal subplots
    titles : list
          It contains titles for subplots
    filename : string
           The name with wich the file will be saved
    path : string
           Path where video will be saved

    Returns
    -------
    It saves a video in the specified directory
    """
    frames = len(data[0])
    fig, axes = plt.subplots(ver, hor, figsize=(15 + 2, ver / hor * 15 + 2),
                             squeeze=False)
    plt.tight_layout()
    lines = []
    axes = axes.flatten()
    for nd, (ax, dat) in enumerate(zip(axes, data)):
        ll = ax.imshow(dat[0, :, :], cmap='RdYlBu', vmin=vmin, vmax=vmax)
        ax.set_ylabel('Width (Log)')
        ax.set_xlabel('Depth (Log)')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        lines.append(ll)
    
    def updates(it, lines, titles):
        if it % 50 == 0:
            print(it)
        for nd, line in enumerate(lines):
            line.set_data(data[nd][it, :, :])
            axes[nd].set_title(titles[nd])
    ani = animation.FuncAnimation(fig, updates, frames=frames,
                                  fargs=(lines, titles), interval=80)
    cax = axes[-1].inset_axes([1.005, 0.05, 0.05, .9],
                              transform=axes[-1].transAxes)
    plt.colorbar(ll, cax=cax, ax=axes[-1])
    plt.subplots_adjust(.02, 0, 0.93, 1, 0.1, 0.05)
    ani.save(path + '/' + filename + '.mp4', dpi=150)
    plt.close()


def graph_anim_suboptimal(data, loc, ver, hor, titles, filename, path):
    """
    Create a neural network animation.

    Parameters
    ----------
    data : list
           Each element
    loc : list
           Node location
    ver : integer
          Vertical subplots
    hor : integer
          Horizontal subplots
    titles : list
          It contains titles for subplots
    filename : string
           The name with wich the file will be saved
    path : string
           Path where video will be saved

    Returns
    -------
    It saves a video in the specified directory
    """
    frames = len(data[0])
    fig, axes = plt.subplots(ver, hor)
    axes = axes.flatten()
    positions = []
    lines = []
    adjacencies = []
    for j, (dat, p) in enumerate(zip(data, loc)):
        G = dat[0]
        adjacencies.append(G)
        Gn = nx.from_numpy_matrix(dat[0])
        nodes = list(Gn.nodes)
        pos = {}
        ind = 0
        for i, node in enumerate(nodes):
            before = p[ind]
            current = p[ind + 1] - before
            pos[node] = (2 * ind, i - before - (current - 1) / 2)
            if i >= p[ind + 1] - 1:
                ind += 1
        positions.append(pos)
        g = Graph(G, node_layout=pos, node_size=20, arrows=False, ax=axes[j],
                  node_color='#24FF00')
        axes[j].set_title(titles[j], y=.94)
        lines.append(g)
    col_pos = plt.cm.Oranges(np.linspace(0.35, 0.95, 256))
    col_neg = plt.cm.Blues_r(np.linspace(0.05, 0.65, 256))
    all_cols = np.vstack((col_neg, col_pos))
    w_map = colors.LinearSegmentedColormap.from_list('w_map', all_cols)
    def update(it):
        print(it)
        artists = []
        for i, (dat, p) in enumerate(zip(data, pos)):
            for jj, kk in zip(*np.where(dat[it] != 0)):
                w = dat[it][jj, kk]
                lines[i].edge_artists[(jj, kk)].set_facecolor(w_map(w))
                lines[i].edge_artists[(jj, kk)].width = 0.01*np.abs(w)
                lines[i].edge_artists[(jj, kk)]._update_path()
                artists.append(lines[i].edge_artists[(jj, kk)])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=500, blit=True)
    plt.subplots_adjust(0, 0, 1, 0.96, 0.05, 0.03)
    ani.save(path + '/' + filename + '.mp4', dpi=500)
    plt.close()

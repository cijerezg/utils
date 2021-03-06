"""Contain various functions."""
import torch
import numpy as np
import smtplib
import socket
from email.message import EmailMessage
import pdb


def angle_between_vecs(vecs):
    """
    Compute the angle between all vectors and norms.

    Parameters
    ----------
    vecs: list
      Elements contain the vectors.

    Returns
    -------
    array: ndarray
      The array that contains the angle in radians
      between all pairs of vectors.
    array: ndarray
      The array contains the vector norms.
    """
    angles = torch.zeros(len(vecs), len(vecs))
    norms = torch.zeros(len(vecs))
    layers = len(vecs[0])
    for q in range(len(vecs)):
        norm = 0
        for p in range(layers):
            vec = vecs[q][p].flatten()
            if not torch.is_tensor(vec):
                vec = torch.from_numpy(vec)
            aux_norm = torch.dot(vec, vec)
            norm += aux_norm
        norms[q] = torch.sqrt(norm)
    for j in range(len(vecs)):
        for i in range(j + 1, len(vecs)):
            dot = 0
            for p in range(layers):
                vec1 = vecs[j][p].flatten()
                vec2 = vecs[i][p].flatten()
                if not torch.is_tensor(vec1):
                    vec1 = torch.from_numpy(vec1)
                    vec2 = torch.from_numpy(vec2)
                aux_dot = torch.dot(vec1, vec2)
                dot += aux_dot
            dot_prod = dot / (norms[i] * norms[j])
            dot_prod = torch.acos(dot_prod)
            angles[j, i] = dot_prod
    angles = angles.cpu().detach().numpy()
    norms = norms.cpu().detach().numpy()
    return angles, norms


def adjecency_matrix(dataset):
    """
    Compute adjecency matrix for fully connected neural network.

    Parameters
    ---------
    dataset: list
       Each element contains an array and each array should match
       size with the next array (as in NNs). Arrays are of the form
       (m, n), where n is input dimension and m is output dimension

    Returns
    -------
    array: ndarray
      Adjecency matrix of size that has the sum of widhts of all layers
    """
    s = [data.shape[1] for data in dataset]
    s.append(dataset[-1].shape[0])
    total_size = sum(s)
    graph = np.zeros((total_size, total_size))
    s.insert(0, 0)
    s = np.cumsum(np.asarray(s))
    for i, data in enumerate(dataset):
        graph[s[i + 1]:s[i + 2], s[i]:s[i + 1]] = data
        graph[s[i]:s[i + 1], s[i + 1]:s[i + 2]] = np.transpose(data)
    return s, graph


def renormalizaion_3d(ref, data):
    """
    Normalize data to have mean zero and biggest values as ref.

    Parameters
    ----------
    ref: array
      3D array. First axis is shapes (e.g. sphere), second axis is
      points, third axis is dimensions x, y, and z.
    data: array
      4D array. First axis is epochs, second axis is shapes, third
      axis is points, and fourth axis is dimensions x, y, and z.

    Returns
    -------
    array: ndarray
      4D array, same shape as data, but normalized.
    """
    outs_norm = data - np.mean(data, 2)[:, :, np.newaxis, :]
    pt_out = np.ptp(outs_norm, axis=2) / 2
    pt_maxim = np.max(pt_out, 2)
    outs_norm = outs_norm / pt_maxim[:, :, np.newaxis, np.newaxis]
    pt_ref = np.ptp(ref, axis=1) / 2
    outs_norm = outs_norm * pt_ref[:, 0][np.newaxis, :, np.newaxis, np.newaxis]
    return outs_norm


def torus(R, r, points):
    """
    Create a torus with outer radius R and inner radius r.

    Parameters
    ----------
    R: float
      Outer radius (bigger than r).
    r: float
      Inner radius (smllaer than R).
    points: int
      Number of points to use. Actual number of points is points squared.

    Returns
    -------
    array: ndarray
      (points^2, 3) array that contains points of torus
    """
    angle = np.linspace(0, 2 * np.pi, points)
    u, v = np.meshgrid(angle, angle)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    data = np.concatenate(
        (x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), 1)
    return data


def sphere(r, points):
    """
    Create a sphere with radius r.

    Parameters
    ----------
    r: float
      Sphere radius.
    points: int
      Number of points to use. Actual number of points is points squared.

    Returns
    -------
    array: ndarray
      (points^2, 3) array that contains points of sphere
    """
    angle = np.linspace(0, 2 * np.pi, points)
    u, v = np.meshgrid(angle, angle)
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    data = np.concatenate(
        (x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), 1)
    return data


def gradients_analysis(grads):
    """
    Count how deep gradients go and what percentage of width.

    Parameters
    ----------
    grads: list
       Each element contains a list that contains the whole gradients of each
       neural net layer.

    Returns
    -------
    array: ndarray
       An array that contains how deep gradients propagated per step
    array: ndarray
       An array that contains what percentage of paths allow propagation.
    """
    nonzero_layer = []
    percentage = []
    n_layers = len(grads[0])
    grads_copy = grads.copy()
    for grad in grads_copy:
        count = 0
        percs = 0
        for layer in grad:
            layer[np.abs(layer) < 1e-6] = 0
            nonzer = np.count_nonzero(layer)
            per = nonzer/layer.size
            if per != 0:
                count += 1
                percs += per
        nonzero_layer.append(count/n_layers)
        percentage.append(percs/(count+0.00000000001))
    nonzero = np.asarray(nonzero_layer)
    percentage = np.asarray(percentage)
    return nonzero, percentage


def send_notification(script_name):
    """Send email."""
    email = 'notifypythone@gmail.com'
    passwd = '1234abcd?'
    email1 = 'cijerezg@unal.edu.co'

    msg = EmailMessage()
    msg.set_content('Hello')
    msg['Subject'] = 'Your script '+script_name+' is done'

    try:
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.ehlo()
        smtp.starttls()
        smtp.login(email, passwd)
        smtp.sendmail(email, email1, msg.as_string())
        smtp.quit()
    except socket.gaierror:
        print('Network connection error')


def lp_norm(x, p):
    """
    Calculate p norm of vector x.

    Parameters
    ----------
    x: array
      Vector 
    p: integer
      p-norm to be calculated
    """
    return np.power(np.sum(np.abs(np.power(x, p)), axis=1), 1/p)
        

def slider(X1, X2, p, ind=None):
    """
    Slide two arrays and computes lp norms.

    Parameters
    ----------
    X1: array
      Array to slide
    X2: array
      Array to slide
    p: integer
      p-norm to be considered

    Returns
    -------
    array: ndarray
       Matrix with slided values
    """
    if ind is None:
        u, ind = np.unique(X1, return_index=True, axis=0)
    X1 = np.take(X1, ind, axis=0)
    X2 = np.take(X2, ind, axis=0)
    size = X1.shape[0]
    sliders = np.full((size, size), 100.0)
    for i in range(1, X1.shape[0]):
        index1 = np.arange(size-i)
        index2 = index1+i
        if len(X1.shape) == 1 and i == 1:
            X1 = X1[:, np.newaxis]
            X2 = X2[:, np.newaxis]
        X1_aux = X1[i::, :]
        X2_aux = X2[:(size-i), :]
        sliders[index1, index2] = lp_norm(X1_aux+X2_aux, p)
    return sliders, ind

        
def lipschitz_dataset(datasets, p):
    """
    Calculate mean and max Lipschitz constant for dataset.

    Parameters
    ----------
    datasets: list
      Each elements contains a tuple that is a dataset.

    Returns
    -------
    array: ndarray
       Lipshitz constants of dataset
    """
    Lipschitz_C = []
    new_data = []
    for dataset in datasets:
        X, Y = dataset
        sliders_X, ind = slider(X, -X, p, ind=None)
        sliders_Y, _ = slider(Y, -Y, p, ind)
        Ks = sliders_Y/sliders_X
        X = np.take(X, ind, axis=0)
        Y = np.take(Y, ind, axis=0)
        idx = np.argwhere(Ks > 5).flatten()
        idx = set(idx)
        idx = list(set(np.arange(X.shape[0]))-idx)
        X = X[idx, :]
        Y = Y[idx]
        new_data.append((X, Y))
        inds = np.triu_indices(Ks.shape[0], k=1)
        Ks = Ks[inds]
        Ks = np.sort(Ks)
        Lipschitz_C.append(Ks)
    return Lipschitz_C, new_data


def slider_torch_dot(X1, X2):
    """
    Slide two arrays and computes dot product.

    Parameters
    ----------
    X1: array
      Array to slide
    X2: array
      Array to slide
    p: integer
      p-norm to be considered

    Returns
    -------
    array: ndarray
       Matrix with slided values
    """
    size = X1.shape[0]
    sliders = torch.full((size, size), 0.0)
    sliders = sliders.double().to(torch.device('cuda'))
    for i in range(1, size):
        ind1 = torch.arange(size-i)
        ind2 = ind1+i
        X1_aux = X1[i::, :]
        X2_aux = X2[:(size-i), :]
        sliders[ind1, ind2] = torch.einsum('ij,ij->i', X1_aux, X2_aux)
    return sliders


def per_sample_angles_grad(data):
    """
    Compute angles between samples and select biggest.

    Parameters
    ----------
    data: list
       Each element contains a layer

    Returns
    list: tuple
       First element contains indices; second element angles.
    """
    eps = 1e-7
    samples = data[0].shape[0]
    flat_data = torch.empty(samples, 0).to(torch.device('cuda'))
    for weights in data:
        flat_weight = weights.reshape(samples, -1)
        flat_data = torch.cat((flat_data, flat_weight), 1)
    norms = torch.einsum('ij,ij->i', flat_data, flat_data)
    norms = torch.sqrt(norms)
    norms = torch.einsum('i,j->ij', norms, norms)
    flat_data = slider_torch_dot(flat_data, flat_data)
    angle = flat_data/norms
    angle = torch.clamp(angle, -1+eps, 1-eps)
    norms, flat_data = None, None
    angle = torch.acos(angle)
    angle = angle.cpu().detach().numpy()
    inds = np.argwhere(angle > 31*np.pi/32)
    return inds

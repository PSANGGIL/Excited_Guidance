#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dgl
import dgl.function as dgl_fn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from scipy.optimize import linear_sum_assignment
from torch import einsum
from torch.distributions import Categorical, Exponential
from torch.nn.functional import one_hot, softmax
from torch.optim import Optimizer
from torch_scatter import segment_csr


def _safe_feature_loss(
    loss_fn: nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | float,
    *,
    time_scaled_loss: bool,
    ctmc_categorical: bool,
) -> torch.Tensor:
    """Reduce a feature loss without NaN for an all-ignored CTMC batch.

    CTMC categorical entries already unmasked at the sampled time use the
    CrossEntropy ignore index -100. If a whole modality is ignored, PyTorch's
    mean cross-entropy has zero valid elements and returns NaN. Such a modality
    must contribute a differentiable zero instead.
    """
    if ctmc_categorical:
        if target.ndim != 1:
            raise ValueError(
                "CTMC categorical targets must be integer class indices. "
                "Use target_blur=0.0 with CTMC parameterization."
            )
        valid = target.ne(-100)
        if not bool(valid.any()):
            return prediction.sum() * 0.0
        if time_scaled_loss:
            element_loss = loss_fn(prediction, target)[valid]
            if torch.is_tensor(weight):
                weight = weight[valid]
                while weight.ndim < element_loss.ndim:
                    weight = weight.unsqueeze(-1)
            return (element_loss * weight).mean()
        return loss_fn(prediction[valid], target[valid])

    element_loss = loss_fn(prediction, target)
    if torch.is_tensor(weight):
        while weight.ndim < element_loss.ndim:
            weight = weight.unsqueeze(-1)
    output = element_loss * weight
    return output.mean() if time_scaled_loss else output

# ========================================================================================
# Dirichlet-flow utilities (original molguidance/utils/dirflow.py)
# ========================================================================================

# this code is adapted from the DirichiletFlow paper

class DirichletConditionalFlow:
    def __init__(self, K=20, alpha_min=1, alpha_max=100, alpha_spacing=0.01):
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha):
        out1 = scipy.special.beta(alpha, self.K - 1)

        denom = (1 - bs) ** (self.K - 1)
        really_small_mask = np.isclose(denom, 0, atol=1.0e-8)
        out2 = np.where(~really_small_mask, out1 / denom, 0)

        denom = bs**(alpha - 1)
        really_small_mask = np.isclose(denom, 0, atol=1.0e-8)
        out = np.where(~really_small_mask, out2 /denom, 0)

        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)
        final = interp * out

        return final

def simplex_proj(x):
    """Algorithm from https://arxiv.org/abs/1309.1541 Weiran Wang, Miguel Á. Carreira-Perpiñán"""
    # seq has shape (batch_size, sequence_length, alphabet_size)
    Y = x.reshape(-1, x.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(x.shape)

# ========================================================================================
# Data/graph utilities (original molguidance/data_processing/utils.py)
# ========================================================================================

def build_edge_idxs(n_atoms: int):
    """Builds an array of edge indices for a molecule with n_atoms.

    The edge indicies are constructed such that the upper-triangle of the adjacency matrix is traversed before the lower triangle.
    Much of our infrastructure relies on this particular ordering of edge indicies within our graph objects.
    """
    # get upper triangle of adjacency matrix
    upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)

    # get lower triangle edges by swapping source and destination of upper_edge_idxs
    lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

    edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
    return edges

def get_upper_edge_mask(g: dgl.DGLGraph):
        """Returns a boolean mask for the edges that lie in the upper triangle of the adjacency matrix for each molecule in the batch."""
        # this algorithm assumes that the edges are ordered such that the upper triangle edges come first, followed by the lower triangle edges for each graph in the batch
        # and then those graph-wise edges are concatenated together
        # you can see that this is indeed how the edges are constructed by inspecting data_processing.dataset.MoleculeDataset.__getitem__
        edges_per_mol = g.batch_num_edges()
        ul_pattern = torch.tensor([1,0]).repeat(g.batch_size).to(g.device)
        n_edges_pattern = (edges_per_mol/2).int().repeat_interleave(2)
        upper_edge_mask = ul_pattern.repeat_interleave(n_edges_pattern).bool()
        return upper_edge_mask

def get_node_batch_idxs(g: dgl.DGLGraph):
    """Returns a tensor of integers indicating which molecule each node belongs to."""
    node_batch_idx = torch.arange(g.batch_size, device=g.device)
    node_batch_idx = node_batch_idx.repeat_interleave(g.batch_num_nodes())
    return node_batch_idx

def get_edge_batch_idxs(g: dgl.DGLGraph):
    """Returns a tensor of integers indicating which molecule each edge belongs to."""
    edge_batch_idx = torch.arange(g.batch_size, device=g.device)
    edge_batch_idx = edge_batch_idx.repeat_interleave(g.batch_num_edges())
    return edge_batch_idx

def get_batch_idxs(g: dgl.DGLGraph):
    """Returns two tensors of integers indicating which molecule each node and edge belongs to."""
    node_batch_idx = get_node_batch_idxs(g)
    edge_batch_idx = get_edge_batch_idxs(g)
    return node_batch_idx, edge_batch_idx

# ========================================================================================
# Prior distributions and alignment (original molguidance/data_processing/priors.py)
# ========================================================================================

def gaussian(n: int, d: int, std: float = 1.0, simplex_center: bool = False):
    """
    Generate a prior feature by sampling from a Gaussian distribution.
    """
    p = torch.randn(n, d) * std

    if simplex_center:
        p = p + 1/d
    return p


def centered_normal_prior(n: int, d: int, std: float = 4.0):
    """
    Generate a prior feature by sampling from a centered normal distribution.
    """
    prior_feat = torch.randn(n, d) * std
    prior_feat = prior_feat - prior_feat.mean(dim=0, keepdim=True)
    return prior_feat

def centered_normal_prior_batched_graph(g: dgl.DGLGraph, node_batch_idx: torch.Tensor, std: float = 4.0):

    n = g.num_nodes()
    prior_sample = torch.randn(n, 3, device=g.device)
    with g.local_scope():
        g.ndata['prior_sample'] = prior_sample
        prior_sample = prior_sample - dgl.readout_nodes(g, feat='prior_sample', op='mean')[node_batch_idx]

    return prior_sample



def barycenter_prior(n: int, d: int, blur: float = 0.0):

    p = torch.ones(n,d) / d

    if blur != 0.0:
        p = p + torch.randn_like(p) * blur
        p = simplex_proj(p)

    return p


def biased_simplex_prior(n, d, vertex_prob: float = 0.75, std: float = 0.2, vertex_idx: int = 0):
    """
    Generate samples from a simplex which are biased towards one category.
    """
    non_zero_weight = (1 - vertex_prob) / (d - 1)
    mu = torch.ones(d)*non_zero_weight
    mu[vertex_idx] = vertex_prob
    simplex_sample = mu.unsqueeze(0) + torch.randn(n, d)*std
    simplex_sample = softmax(simplex_sample/(1/d), dim=1)
    return simplex_sample

def uniform_simplex_prior(n, d):
    """
    Generate samples from a uniform distribution on a simplex.
    """
    exp_dist = Exponential(torch.tensor(1.0))
    sample = exp_dist.sample((n, d))
    sample = sample / sample.sum(dim=1, keepdim=True)
    return sample

def sample_marginal(n: int, d: int, p: torch.Tensor, blur: float = None):
    """
    Sample from the marginal distribution of a categorical variable.
    """
    prior_idxs = torch.multinomial(p, n, replacement=True)
    prior_one_hot = one_hot(prior_idxs, num_classes=d).float()

    if blur is not None:
        prior_one_hot = prior_one_hot + torch.randn_like(prior_one_hot) * blur
        prior_one_hot = softmax(prior_one_hot/(1/d), dim=1)

    return prior_one_hot

def sample_p_c_given_a(n: int, d: int, atom_types: torch.Tensor, p_c_given_a: torch.Tensor, blur: float = None):
    """
    Sample from the conditional distribution of charges given atom type, p(c|a).
    """
    if p_c_given_a.device != atom_types.device:
        p_c_given_a = p_c_given_a.to(atom_types.device)

    atom_type_idxs = atom_types.argmax(dim=1)
    charge_idxs = torch.multinomial(p_c_given_a[atom_type_idxs], 1, replacement=True).squeeze(-1)

    charge_simplex = one_hot(charge_idxs, num_classes=d).float()

    if blur is not None:
        charge_simplex = charge_simplex + torch.randn_like(charge_simplex) * blur
        charge_simplex = softmax(charge_simplex/(1/d), dim=1)

    return charge_simplex

def ctmc_masked_prior(n: int, d: int):
    """
    Sample from a CTMC masked prior. All samples are assigned the mask token at t=0.
    """
    p = torch.full((n,), fill_value=d)
    p = one_hot(p, num_classes=d+1).float()
    return p

def align_prior(prior_feat: torch.Tensor, dst_feat: torch.Tensor, permutation=False, rigid_body=False, n_alignments: int = 1):
    """
    Aligns a prior feature to a destination feature.
    """
    for _ in range(n_alignments):
        if permutation:
            # solve assignment problem
            cost_mat = torch.cdist(dst_feat, prior_feat, p=2)
            _, prior_idx = linear_sum_assignment(cost_mat)

            # reorder prior to according to optimal assignment
            prior_feat = prior_feat[prior_idx]

        if rigid_body:
            # perform rigid alignment
            prior_feat = rigid_alignment(prior_feat, dst_feat)

    return prior_feat

def rigid_alignment(x_0, x_1, pre_centered=False):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Alignment of two point clouds using the Kabsch algorithm.
    Based on: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    """
    d = x_0.shape[1]
    assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape"

    # remove COM from data and record initial COM
    if pre_centered:
        x_0_mean = torch.zeros(1, d)
        x_1_mean = torch.zeros(1, d)
        x_0_c = x_0
        x_1_c = x_1
    else:
        x_0_mean = x_0.mean(dim=0, keepdim=True)
        x_1_mean = x_1.mean(dim=0, keepdim=True)
        x_0_c = x_0 - x_0_mean
        x_1_c = x_1 - x_1_mean

    # Covariance matrix
    H = x_0_c.T.mm(x_1_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    if pre_centered:
        t = torch.zeros(1, d)
    else:
        t = x_1_mean - R.mm(x_0_mean.T).T # has shape (1, D)

    # apply rotation to x_0_c
    x_0_aligned = x_0_c.mm(R.T)

    # move x_0_aligned to its original frame
    x_0_aligned = x_0_aligned + x_0_mean

    # apply the translation
    x_0_aligned = x_0_aligned + t

    return x_0_aligned

def batched_rigid_alignment(x_0, x_1, pre_centered=False):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Alignment of two point clouds using the Kabsch algorithm.
    Based on: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    """
    print('WARNING: batched_rigid_alignment is currently broken (gives incorrect results)')
    assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape"

    if len(x_0.shape) == 2:
        n, d = x_0.shape
        b = 1
        x_0 = x_0.unsqueeze(0)
        x_1 = x_1.unsqueeze(0)

    elif len(x_0.shape) == 3:
        b, n, d = x_0.shape

    # remove COM from data and record initial COM
    if pre_centered:
        x_0_mean = torch.zeros(b, 1, d)
        x_1_mean = torch.zeros(b, 1, d)
        x_0_c = x_0
        x_1_c = x_1
    else:
        x_0_mean = x_0.mean(dim=1, keepdim=True)
        x_1_mean = x_1.mean(dim=1, keepdim=True)
        x_0_c = x_0 - x_0_mean
        x_1_c = x_1 - x_1_mean

    # Covariance matrix
    # x_0_c has shape (b, n, d) as does x_1_c
    # H shold have shape (b, d, d)
    # below is the line for the unbatched version, followed by the batched version
    # H = x_0_c.T.mm(x_1_c)
    H = torch.einsum('bnd,bnm->bdm', x_0_c, x_1_c)

    U, S, V = torch.svd(H)
    # Rotation matrix
    # U and V both have shape (b, d, d)
    # R should have shape (b, d, d)
    # below is the line for the unbatched version, followed by the batched version
    # R = V.mm(U.T)
    R = torch.einsum('bxy,bjk->bxj', V, U)

    # Translation vector
    if pre_centered:
        t = torch.zeros(b, 1, d)
    else:
        # R has shape (b, d, d)
        # x_0_mean has shape (b, 1, d)
        # t = x_1_mean - R.mm(x_0_mean.T).T # has shape (b, 1, D)
        t = x_1_mean - torch.einsum('bxy,bjk->bjy', R, x_0_mean)


    # apply rotation to x_0_c
    # x_0_c has shape (b, n, d)
    # R has shape (b, d, d)
    # x_0_aligned should have shape (b, n, d)
    # below is the line for the unbatched version, followed by the batched version
    # x_0_aligned = x_0_c.mm(R.T)
    x_0_aligned = torch.einsum('bxy,bjk->bxk', x_0_c, R)

    # move x_0_aligned to its original frame
    x_0_aligned = x_0_aligned + x_0_mean

    # apply the translation
    x_0_aligned = x_0_aligned + t

    return x_0_aligned



train_prior_register = {
    'centered-normal': centered_normal_prior,
    'uniform-simplex': uniform_simplex_prior,
    'biased-simplex': biased_simplex_prior,
    'marginal': sample_marginal,
    'c-given-a': sample_p_c_given_a,
    'gaussian': gaussian,
    'barycenter': barycenter_prior,
    'ctmc': ctmc_masked_prior
}

inference_prior_register = {
    'centered-normal': centered_normal_prior_batched_graph,
    'uniform-simplex': uniform_simplex_prior,
    'biased-simplex': biased_simplex_prior,
    'marginal': sample_marginal,
    'c-given-a': sample_p_c_given_a,
    'gaussian': gaussian,
    'barycenter': barycenter_prior,
    'ctmc': ctmc_masked_prior
}

@torch.no_grad()
def coupled_node_prior(dst_dict: dict,
                     prior_config: dict):
    prior_dict = {}

    for feat in dst_dict.keys():

        # get the prior configuration for this feature
        feat_prior_config = prior_config[feat]

        # get destination features (t=1)
        dst_feat = dst_dict[feat]

        # sample prior
        prior_fn = train_prior_register[feat_prior_config['type']]
        n, d = dst_feat.shape
        args = [n,d]

        # if sampling the charges conditioned on atom type, we need to pass the atom types to the prior function
        # note that this behavior is dependent on "a" being encountered in this loop before "c"
        if feat == 'c' and feat_prior_config['type'] == 'c-given-a':
            args.append(prior_dict['a'])

        prior_feat = prior_fn(*args, **feat_prior_config['kwargs'])

        # align prior to destination if necessary
        if feat_prior_config['align']:

            if feat == 'x':
                rigid_body = True
            else:
                rigid_body = False

            prior_feat = align_prior(prior_feat, dst_feat, permutation=True, rigid_body=rigid_body)

        prior_dict[feat] = prior_feat

    return prior_dict

def edge_prior(upper_edge_mask: torch.Tensor, edge_prior_config: dict):

    n_upper_edges = upper_edge_mask.sum().item()
    prior_fn = train_prior_register[edge_prior_config['type']]
    upper_edge_prior = prior_fn(n_upper_edges, 5, **edge_prior_config['kwargs'])

    edge_prior = torch.zeros(upper_edge_mask.shape[0], upper_edge_prior.shape[1])
    edge_prior[upper_edge_mask] = upper_edge_prior
    edge_prior[~upper_edge_mask] = upper_edge_prior
    return edge_prior

# ========================================================================================
# CTMC utilities (original molguidance/utils/ctmc_utils.py)
# ========================================================================================

def purity_sampling(xt, x1, x1_probs, unmask_prob, mask_index, batch_size, batch_num_nodes, node_batch_idx, hc_thresh, device):

    masked_nodes = xt == mask_index # mask of which nodes are currently unmasked
    purities = x1_probs.max(-1)[0] # the highest probability of any category for each node

    hc_mask = purities >= hc_thresh # mask of which nodes are high-confidence
    hc_mask = hc_mask * masked_nodes # only consider nodes that are currently masked

    # compute the number of hc nodes in each graph in the batch
    indptr = torch.zeros(batch_size+1, device=device, dtype=torch.long)
    indptr[1:] = batch_num_nodes.cumsum(0)
    hc_nodes_per_graph = segment_csr(hc_mask.long(), indptr) # has shape (batch_size,)

    # compute the number of masked nodes in each graph in the batch
    masked_nodes_per_graph = segment_csr(masked_nodes.long(), indptr) # has shape (batch_size,)

    # compute max value of ph for each graph in the batch
    ph_max = unmask_prob*masked_nodes_per_graph / hc_nodes_per_graph
    ph_max[ hc_nodes_per_graph == 0 ] = torch.inf

    # compute ph and pl for each graph in the batch
    ph = torch.minimum(ph_max, torch.full_like(ph_max, 1.0)) # bernoulli trial probability of high confidence nodes in each graph
    pl = (unmask_prob*masked_nodes_per_graph - ph*hc_nodes_per_graph) / (masked_nodes_per_graph - hc_nodes_per_graph) # bernoulli trial probability of low confidence nodes in each graph

    # construct a tensor containing the unmask probability for every node
    node_unmask_prob = torch.zeros_like(xt).float()
    node_unmask_prob[hc_mask] = ph[node_batch_idx[hc_mask]]
    lc_mask = (purities < hc_thresh) * masked_nodes # nodes which are currently masked and low-confidence
    node_unmask_prob[lc_mask] = pl[node_batch_idx[lc_mask]]

    will_unmask = torch.rand(xt.shape[0], device=device) < node_unmask_prob # sample nodes to unmask
    return will_unmask

# ========================================================================================
# GVP architecture (original molguidance/models/gvp.py)
# ========================================================================================

# helper functions
def exists(val):
    return val is not None

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

# the classes GVP, GVPDropout, and GVPLayerNorm are taken from lucidrains' geometric-vector-perceptron repository
# https://github.com/lucidrains/geometric-vector-perceptron/tree/main
# some adaptations have been made to these classes to make them more consistent with the original GVP paper/implementation
# specifically, using _norm_no_nan instead of torch's built in norm function, and the weight intialiation scheme for Wh and Wu

def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    device = D.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

class GVP(nn.Module):
    def __init__(
        self,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        n_cp_feats = 0, # number of cross-product features added to hidden vector features
        hidden_vectors = None,
        feats_activation = nn.SiLU(),
        vectors_activation = nn.Sigmoid(),
        vector_gating = True,
        xavier_init = False
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in
        self.n_cp_feats = n_cp_feats

        self.dim_vectors_out = dim_vectors_out
        dim_h = max(dim_vectors_in, dim_vectors_out) if hidden_vectors is None else hidden_vectors

        # create Wh matrix
        wh_k = 1/math.sqrt(dim_vectors_in)
        self.Wh = torch.zeros(dim_vectors_in, dim_h, dtype=torch.float32).uniform_(-wh_k, wh_k)
        self.Wh = nn.Parameter(self.Wh)

        # create Wcp matrix if we are using cross-product features
        if n_cp_feats > 0:
            wcp_k = 1/math.sqrt(dim_vectors_in)
            self.Wcp = torch.zeros(dim_vectors_in, n_cp_feats*2, dtype=torch.float32).uniform_(-wcp_k, wcp_k)
            self.Wcp = nn.Parameter(self.Wcp)



        # create Wu matrix
        if n_cp_feats > 0: # the number of vector features going into Wu is increased by n_cp_feats if we are using cross-product features
            wu_in_dim = dim_h + n_cp_feats
        else:
            wu_in_dim = dim_h
        wu_k = 1/math.sqrt(wu_in_dim)
        self.Wu = torch.zeros(wu_in_dim, dim_vectors_out, dtype=torch.float32).uniform_(-wu_k, wu_k)
        self.Wu = nn.Parameter(self.Wu)

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + n_cp_feats + dim_feats_in, dim_feats_out),
            feats_activation
        )

        # branching logic to use old GVP, or GVP with vector gating
        if vector_gating:
            self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out)
            if xavier_init:
                nn.init.xavier_uniform_(self.scalar_to_vector_gates.weight, gain=1)
                nn.init.constant_(self.scalar_to_vector_gates.bias, 0)
        else:
            self.scalar_to_vector_gates = None

        # self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out) if vector_gating else None

    def forward(self, data):
        feats, vectors = data
        b, n, _, v, c  = *feats.shape, *vectors.shape

        # feats has shape (batch_size, n_feats)
        # vectors has shape (batch_size, n_vectors, 3)

        assert c == 3 and v == self.dim_vectors_in, 'vectors have wrong dimensions'
        assert n == self.dim_feats_in, 'scalar features have wrong dimensions'

        Vh = einsum('b v c, v h -> b h c', vectors, self.Wh) # has shape (batch_size, dim_h, 3)

        # if we are including cross-product features, compute them here
        if self.n_cp_feats > 0:
            # convert dim_vectors_in vectors to n_cp_feats*2 vectors
            Vcp = einsum('b v c, v p -> b p c', vectors, self.Wcp) # has shape (batch_size, n_cp_feats*2, 3)
            # split the n_cp_feats*2 vectors into two sets of n_cp_feats vectors
            cp_src, cp_dst = torch.split(Vcp, self.n_cp_feats, dim=1) # each has shape (batch_size, n_cp_feats, 3)
            # take the cross product of the two sets of vectors
            cp = torch.linalg.cross(cp_src, cp_dst, dim=-1) # has shape (batch_size, n_cp_feats, 3)

            # add the cross product features to the hidden vector features
            Vh = torch.cat((Vh, cp), dim=1) # has shape (batch_size, dim_h + n_cp_feats, 3)

        Vu = einsum('b h c, h u -> b u c', Vh, self.Wu) # has shape (batch_size, dim_vectors_out, 3)

        sh = _norm_no_nan(Vh)

        s = torch.cat((feats, sh), dim = 1)

        feats_out = self.to_feats_out(s)

        if exists(self.scalar_to_vector_gates):
            gating = self.scalar_to_vector_gates(feats_out)
            gating = gating.unsqueeze(dim = -1)
        else:
            gating = _norm_no_nan(Vu)

        vectors_out = self.vectors_activation(gating) * Vu

        # if torch.isnan(feats_out).any() or torch.isnan(vectors_out).any():
        #     raise ValueError("NaNs in GVP forward pass")

        return (feats_out, vectors_out)

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class GVPDropout(nn.Module):
    """ Separate dropout for scalars and vectors. """
    def __init__(self, rate):
        super().__init__()
        self.vector_dropout = _VDropout(rate)
        self.feat_dropout = nn.Dropout(rate)

    def forward(self, feats, vectors):
        return self.feat_dropout(feats), self.vector_dropout(vectors)


class GVPLayerNorm(nn.Module):
    """ Normal layer norm for scalars, nontrainable norm for vectors. """
    def __init__(self, feats_h_size, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.feat_norm = nn.LayerNorm(feats_h_size)

    def forward(self, feats, vectors):

        normed_feats = self.feat_norm(feats)

        vn = _norm_no_nan(vectors, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True) + self.eps ) + self.eps
        normed_vectors = vectors / vn
        return normed_feats, normed_vectors



class GVPConv(nn.Module):

    """GVP graph convolution on a homogenous graph."""

    def __init__(self, scalar_size: int = 128, vector_size: int = 16, n_cp_feats: int = 0,
                  scalar_activation=nn.SiLU, vector_activation=nn.Sigmoid,
                  n_message_gvps: int = 1, n_update_gvps: int = 1,
                  use_dst_feats: bool = False, rbf_dmax: float = 20, rbf_dim: int = 16,
                  edge_feat_size: int = 0, coords_range=10, message_norm: Union[float, str] = 10, dropout: float = 0.0,):

        super().__init__()

        # self.edge_type = edge_type
        # self.src_ntype = edge_type[0]
        # self.dst_ntype = edge_type[2]
        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.n_cp_feats = n_cp_feats
        self.scalar_activation = scalar_activation
        self.vector_activation = vector_activation
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.edge_feat_size = edge_feat_size
        self.use_dst_feats = use_dst_feats
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.dropout_rate = dropout
        self.message_norm = message_norm

        # create message passing function
        message_gvps = []
        for i in range(n_message_gvps):

            dim_vectors_in = vector_size
            dim_feats_in = scalar_size

            # on the first layer, there is an extra edge vector for the displacement vector between the two node positions
            if i == 0:
                dim_vectors_in += 1
                dim_feats_in += rbf_dim + edge_feat_size

            # if this is the first layer and we are using destination node features to compute messages, add them to the input dimensions
            if use_dst_feats and i == 0:
                dim_vectors_in += vector_size
                dim_feats_in += scalar_size

            message_gvps.append(
                GVP(dim_vectors_in=dim_vectors_in,
                    dim_vectors_out=vector_size,
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=dim_feats_in,
                    dim_feats_out=scalar_size,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                    vector_gating=True)
            )
        self.edge_message = nn.Sequential(*message_gvps)

        # create update function
        update_gvps = []
        for i in range(n_update_gvps):
            update_gvps.append(
                GVP(dim_vectors_in=vector_size,
                    dim_vectors_out=vector_size,
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=scalar_size,
                    dim_feats_out=scalar_size,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                    vector_gating=True)
            )
        self.node_update = nn.Sequential(*update_gvps)

        self.dropout = GVPDropout(self.dropout_rate)
        self.message_layer_norm = GVPLayerNorm(self.scalar_size)
        self.update_layer_norm = GVPLayerNorm(self.scalar_size)

        if isinstance(self.message_norm, str) and self.message_norm not in ['mean', 'sum']:
            raise ValueError(f"message_norm must be either 'mean', 'sum', or a number, got {self.message_norm}")
        else:
            assert isinstance(self.message_norm, (float, int)), "message_norm must be either 'mean', 'sum', or a number"

        if self.message_norm == 'mean':
            self.agg_func = dgl_fn.mean
        else:
            self.agg_func = dgl_fn.sum

    def forward(self, g: dgl.DGLGraph,
                scalar_feats: torch.Tensor,
                coord_feats: torch.Tensor,
                vec_feats: torch.Tensor,
                edge_feats: torch.Tensor = None,
                x_diff: torch.Tensor = None,
                d: torch.Tensor = None):
        # vec_feat has shape (n_nodes, n_vectors, 3)

        with g.local_scope():

            g.ndata['h'] = scalar_feats
            g.ndata['x'] = coord_feats
            g.ndata['v'] = vec_feats

            if x_diff is not None and d is not None:
                g.edata['x_diff'] = x_diff
                g.edata['d'] = d

            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feats is not None, "Edge features must be provided."
                g.edata["a"] = edge_feats



            # normalize x_diff and compute rbf embedding of edge distance
            # dij = torch.norm(g.edges[self.edge_type].data['x_diff'], dim=-1, keepdim=True)
            if 'x_diff' not in g.edata:
                # get vectors between node positions
                g.apply_edges(dgl_fn.u_sub_v("x", "x", "x_diff"))
                dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
                g.edata['x_diff'] = g.edata['x_diff'] / dij
                g.edata['d'] = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)

            # compute messages on every edge
            g.apply_edges(self.message)

            # aggregate messages from every edge
            g.update_all(dgl_fn.copy_e("scalar_msg", "m"), self.agg_func("m", "scalar_msg"))
            g.update_all(dgl_fn.copy_e("vec_msg", "m"), self.agg_func("m", "vec_msg"))

            # get aggregated scalar and vector messages
            if isinstance(self.message_norm, str):
                z = 1
            else:
                z = self.message_norm

            scalar_msg = g.ndata["scalar_msg"] / z
            vec_msg = g.ndata["vec_msg"] / z

            # dropout scalar and vector messages
            scalar_msg, vec_msg = self.dropout(scalar_msg, vec_msg)

            # update scalar and vector features, apply layernorm
            scalar_feat = g.ndata['h'] + scalar_msg
            vec_feat = g.ndata['v'] + vec_msg
            scalar_feat, vec_feat = self.message_layer_norm(scalar_feat, vec_feat)

            # apply node update function, apply dropout to residuals, apply layernorm
            scalar_residual, vec_residual = self.node_update((scalar_feat, vec_feat))
            scalar_residual, vec_residual = self.dropout(scalar_residual, vec_residual)
            scalar_feat = scalar_feat + scalar_residual
            vec_feat = vec_feat + vec_residual
            scalar_feat, vec_feat = self.update_layer_norm(scalar_feat, vec_feat)

        return scalar_feat, vec_feat

    def message(self, edges):

        # concatenate x_diff and v on every edge to produce vector features
        vec_feats = [ edges.data["x_diff"].unsqueeze(1), edges.src["v"] ]
        if self.use_dst_feats:
            vec_feats.append(edges.dst["v"])
        vec_feats = torch.cat(vec_feats, dim=1)

        # create scalar features
        scalar_feats = [ edges.src['h'], edges.data['d'] ]
        if self.edge_feat_size > 0:
            scalar_feats.append(edges.data['a'])

        if self.use_dst_feats:
            scalar_feats.append(edges.dst['h'])

        scalar_feats = torch.cat(scalar_feats, dim=1)

        scalar_message, vector_message = self.edge_message((scalar_feats, vec_feats))

        return {"scalar_msg": scalar_message, "vec_msg": vector_message}

# ========================================================================================
# Interpolant scheduler (original molguidance/models/interpolant_scheduler.py)
# ========================================================================================

class InterpolantScheduler(nn.Module):

    supported_schedule_types = ['cosine', 'linear']

    def __init__(self, canonical_feat_order: str, schedule_type: Union[str, Dict[str, str]] = 'cosine', cosine_params: dict | None = None):
        super().__init__()

        cosine_params = copy.deepcopy(cosine_params or {})
        self.feats = list(canonical_feat_order)
        self.n_feats = len(self.feats)

        # check that schedule_type is a string or a dictionary
        if not isinstance(schedule_type, (str, dict)):
            raise ValueError('schedule_type must be a string or a dictionary')

        # if it is a string, assign the same schedule_type to all features
        if isinstance(schedule_type, str):
            if schedule_type not in self.supported_schedule_types:
                raise ValueError(f'unsupported schedule_type: {schedule_type}')
            self.schedule_dict = {
                feat: schedule_type for feat in self.feats
            }
        else:
            # schedule_type is a dictionary specifying the schedule_type for each feature
            for feat in self.feats:
                if feat not in schedule_type:
                    raise ValueError(f'must specify schedule_type for feature {feat}')

            self.schedule_dict = schedule_type

        # if schedule_type == 'cosine':
        #     self.alpha_t = self.cosine_alpha_t
        #     self.alpha_t_prime = self.cosine_alpha_t_prime
        # elif schedule_type == 'linear':
        #     self.alpha_t = self.linear_alpha_t
        #     self.alpha_t_prime = self.linear_alpha_t_prime
        # else:
        #     raise NotImplementedError(f'unsupported schedule_type: {schedule_type}')


        # for features which have a cosine schedule, check that the parameter "nu" is provided
        for feat, schedule_type in self.schedule_dict.items():
            if schedule_type == 'cosine' and feat not in cosine_params:
                raise ValueError(f'must specify cosine_params for feature {feat}')

        # get a list of unique schedule types which are used
        self.schedule_types = list(set( self.schedule_dict.values() ))

        # if we are using a cosine schedule, convert all of the cosine_params to torch tensors
        if 'cosine' in self.schedule_types:
            for feat in cosine_params:
                cosine_params[feat] = torch.tensor(cosine_params[feat]).unsqueeze(0)

        # save the cosine_params as an attribute
        self.cosine_params = cosine_params

        self.device = None

        self.clamp_t = True



    def update_device(self, t):
        if 'cosine' in self.schedule_types and t.device != self.device:
            for key in self.cosine_params:
                self.cosine_params[key] = self.cosine_params[key].to(t.device)
            self.device = t.device

    def interpolant_weights(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the weights for x_0 and x_1 in the interpolation between x_0 and x_1.
        """
        # t has shape (n_timepoints,)
        # returns a tuple of 2 tensors of shape (n_timepoints, n_feats)
        # the tensor at index 0 is the weight for x_0
        # the tensor at index 1 is the weight for x_1

        self.update_device(t)

        alpha_t = self.alpha_t(t)
        weights = (1 - alpha_t, alpha_t)
        return weights

    def loss_weights(self, t: torch.Tensor):
        alpha_t = self.alpha_t(t)
        # alpha_t_prime = self.alpha_t_prime(t)
        # weights = alpha_t_prime/(1 - alpha_t + 1e-5)
        weights = alpha_t/(1 - alpha_t)

        # clamp the weights with a minimum of 0.05 and a maximum of 1.5
        weights = torch.clamp(weights, min=0.05, max=1.5)
        return weights

    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:

        self.update_device(t)

        per_feat_alpha = []
        for feat in self.feats:
            schedule_type = self.schedule_dict[feat]
            if schedule_type == 'cosine':
                alpha_t = self.cosine_alpha_t(t, nu=self.cosine_params[feat])
            elif schedule_type == 'linear':
                alpha_t = self.linear_alpha_t(t)

            per_feat_alpha.append(alpha_t)

        alpha_t = torch.cat(per_feat_alpha, dim=1)
        return alpha_t

    def alpha_t_prime(self, t: torch.Tensor) -> torch.Tensor:
        self.update_device(t)

        per_feat_alpha_prime = []
        for feat in self.feats:
            schedule_type = self.schedule_dict[feat]
            if schedule_type == 'cosine':
                alpha_t_prime = self.cosine_alpha_t_prime(t, nu=self.cosine_params[feat])
            elif schedule_type == 'linear':
                alpha_t_prime = self.linear_alpha_t_prime(t)

            per_feat_alpha_prime.append(alpha_t_prime)

        alpha_t_prime = torch.cat(per_feat_alpha_prime, dim=1)
        return alpha_t_prime


    def cosine_alpha_t(self, t: torch.Tensor, nu: torch.Tensor) -> Dict[str, torch.Tensor]:
        # t has shape (n_timepoints,)
        # alpha_t has shape (n_timepoints, n_feats) containing the alpha_t for each feature
        t = t.unsqueeze(-1)
        alpha_t = 1 - torch.cos(torch.pi*0.5*torch.pow(t, nu)).square()
        return alpha_t

    def cosine_alpha_t_prime(self, t: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:

        if self.clamp_t:
            t = torch.clamp_(t, min=1e-9)

        t = t.unsqueeze(-1)
        sin_input = torch.pi*torch.pow(t, nu)
        alpha_t_prime = torch.pi*0.5*torch.sin(sin_input)*nu*torch.pow(t, nu-1)
        return alpha_t_prime

    def linear_alpha_t(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t = t.unsqueeze(-1)
        return alpha_t

    def linear_alpha_t_prime(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t_prime = torch.ones_like(t).unsqueeze(-1)
        return alpha_t_prime

# ========================================================================================
# Learning-rate scheduler (original molguidance/models/lr_scheduler.py)
# ========================================================================================

# TODO: refacotr scheduler to have a minimium learning rate when doing restarts

class LRScheduler:

    def __init__(self,
                 model: FlowMol,
                 optimizer: Optimizer,
                 base_lr: float,
                 weight_decay: float = 0,
                 warmup_length: float = 0,
                 restart_interval: float = 0,
                 restart_type: str = None):

        self.model = model
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.restart_interval = restart_interval
        self.restart_type = restart_type
        self.warmup_length = warmup_length

        self.restart_marker = self.warmup_length

        if restart_interval != 0 and restart_type is None:
            raise ValueError('must specify a restart type if restart_interval is not 0')

        if self.restart_type == 'linear':
            self.restart_fn = self.linear_restart
        elif self.restart_type == 'cosine':
            self.restart_fn = self.cosine_restart
        else:
            raise NotImplementedError

    def step_lr(self, epoch_exact):

        if epoch_exact <= self.warmup_length and self.warmup_length != 0:
            self.optimizer.param_groups[0]['lr'] = self.base_lr*epoch_exact/self.warmup_length
            return

        if self.restart_interval == 0:
            return

        # assuming we are out of the warmup phase and we are now doing restarts
        epochs_into_interval = epoch_exact - self.restart_marker
        if epochs_into_interval < self.restart_interval: # if we are within a restart interval
            self.optimizer.param_groups[0]['lr'] = self.restart_fn(epochs_into_interval)
        elif epochs_into_interval >= self.restart_interval:
            self.restart_marker = epoch_exact
            self.optimizer.param_groups[0]['lr'] = self.restart_fn(0)
            # TODO: save model on restart
            # model_file = self.output_dir / f'model_on_restart_{epoch_exact:.0f}.pt'
            # save_model(self.model, model_file)


    def linear_restart(self, epochs_into_interval):
        new_lr = -1.0*self.base_lr*epochs_into_interval/self.restart_interval + self.base_lr
        return new_lr

    def cosine_restart(self, epochs_into_interval):
        new_lr = 0.5*self.base_lr*(1+np.cos(epochs_into_interval*np.pi/self.restart_interval))
        return new_lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

# ========================================================================================
# Property embeddings (original molguidance/models/property_embeddings.py)
# ========================================================================================

class PropertyEmbedder(nn.Module):
    def __init__(self, input_dim: int = 1,
                 embedding_dim: int = 128,
                 start: float = 0.0246,
                 stop: float = 0.6221,
                 n_gaussians: int = 5,
                 use_activation: bool = True):
        """
        start (float): is min value of the property, default is for gap
        stop (float): is max value of the property, default is for gap
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # First layer
        layers = [nn.Linear(n_gaussians, embedding_dim)]
        if use_activation:
            layers.append(nn.SiLU())
        layers.append(nn.Linear(embedding_dim, embedding_dim))

        self.mlp = nn.Sequential(*layers)

        # Gaussian expansion layer
        self.gaussian_expansion = GaussianExpansion(start=start, stop=stop, n_gaussians=n_gaussians, trainable=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Property values tensor of shape (batch_size, 1)
        Returns:
            Embedded tensor of shape (batch_size, embedding_dim)
        """
        expanded_x = self.gaussian_expansion(x)
        return self.mlp(expanded_x)

class GaussianExpansion(nn.Module):
    # GaussianExpansion class code reference: https://github.com/atomistic-machine-learning/cG-SchNet/blob/main/nn_classes.py#L616
    r"""Expansion layer using a set of Gaussian functions.

    Args:
        start (float): center of first Gaussian function, :math:`\mu_0`.
        stop (float): center of last Gaussian function, :math:`\mu_{N_g}`.
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`
            (default: 50).
        trainable (bool, optional): if True, widths and offset of Gaussian functions
            are adjusted during training process (default: False).
        widths (float, optional): width value of Gaussian functions (provide None to
            set the width to the distance between two centers :math:`\mu`, default:
            None).

    """

    def __init__(self, start, stop, n_gaussians=50, trainable=False,
                 width=None):
        super(GaussianExpansion, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = torch.FloatTensor((offset[1] - offset[0]) *
                                       torch.ones_like(offset))
        else:
            widths = torch.FloatTensor(width * torch.ones_like(offset))
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, property):
        """Compute expanded gaussian property values.

        Args:
            property (torch.Tensor): property values of (N_b x 1) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_g) shape.

        """
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(self.widths, 2)[None, :]
        # Use advanced indexing to compute the individual components
        diff = property - self.offsets[None, :]
        # compute expanded property values
        return torch.exp(coeff * torch.pow(diff, 2))

# ========================================================================================
# Molecule construction (original molguidance/analysis/molecule_builder.py)
# ========================================================================================

bond_type_map = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC, None] # last bond type is for masked bonds

bond_type_to_idx = { bond_type:idx for idx, bond_type in enumerate(bond_type_map)}
bond_type_to_idx[None] = 0


class SampledMolecule:

    def __init__(self, g: dgl.DGLGraph,
        atom_type_map: List[str],
        traj_frames: Dict[str, torch.Tensor] = None,
        ctmc_mol: bool = False, # whether the molecule was sampled from a CTMC model. Important because one-hot encodings will contain a mask token.
        exclude_charges: bool = False,
        align_traj: bool = True,
        build_xt_traj=True,
        build_ep_traj=True,):
        """Represents a molecule sampled from a model. Converts the DGL graph to an rdkit molecule and keeps all associated information."""

        atom_type_map = list(atom_type_map) # create a shallow copy of the atom type map so that we don't modify the original

        self.exclude_charges = exclude_charges
        self.align_traj = align_traj
        self.ctmc_mol = ctmc_mol

        if ctmc_mol:
            atom_type_map.append('Se') # masked molecules will show up as selenium

        # save the graph
        self.g = g

        self.positions, self.atom_types, self.atom_charges, self.bond_types, self.bond_src_idxs, self.bond_dst_idxs = extract_moldata_from_graph(
            g,
            atom_type_map,
            exclude_charges=exclude_charges,
            ctmc_mol=self.ctmc_mol)

        self.atom_type_map = atom_type_map
        self.num_atoms = g.num_nodes()
        self.num_atom_types = len(atom_type_map)

        # build rdkit molecule
        self.rdkit_mol = self.build_molecule()

        # compute valencies on every atom
        self.valencies = self.compute_valencies()

        # build trajectory molecules
        self.traj_frames = traj_frames
        if traj_frames is not None:

            if build_xt_traj:
                self.traj_mols = self.process_traj_frames(traj_frames) # convert frames into a list of rdkit molecules

            # construct molecules for endpoint trajectory
            if 'x_1_pred' in traj_frames and build_ep_traj:
                self.ep_traj_mols = self.process_traj_frames(traj_frames, ep_traj=True)

    @classmethod
    def from_rdkit_mol(cls, mol: Chem.Mol, atom_type_map: List[str] = None):
        """Creates a SampledMolecule from an rdkit molecule."""

        if atom_type_map is None:
            atom_types = [ atom.GetSymbol() for atom in mol.GetAtoms() ]
            atom_type_map = list(set(atom_types))

        atom_type_to_idx = {atom_type: i for i, atom_type in enumerate(atom_type_map)}

        # get positions
        positions = mol.GetConformer().GetPositions()
        positions = torch.from_numpy(positions)

        # get atom elements as a string
        # atom_types_str = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        atom_types_idx = torch.zeros(mol.GetNumAtoms()).long()
        atom_charges = torch.zeros_like(atom_types_idx)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_types_idx[i] = atom_type_to_idx[atom.GetSymbol()]
            atom_charges[i] = atom.GetFormalCharge()

        # get one-hot encoded of existing bonds only (no non-existing bonds)
        adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
        edge_index = adj.triu().nonzero().contiguous() # upper triangular portion of adjacency matrix

        bond_types = adj[edge_index[:, 0], edge_index[:, 1]]
        bond_src_idxs = edge_index[:, 0]
        bond_dst_idxs = edge_index[:, 1]
        bond_types[bond_types == 1.5] = 4
        edge_attr = bond_types.type(torch.int32)

        # create graph
        g = dgl.graph((bond_src_idxs, bond_dst_idxs), num_nodes=mol.GetNumAtoms())

        # add data to the graph
        g.ndata['x_1'] = positions
        g.ndata['a_1'] = one_hot(atom_types_idx.long(), num_classes=len(atom_type_map)).float()
        g.ndata['c_1'] = one_hot(atom_charges.long() + 2, num_classes=6).float()
        g.edata['e_1'] = one_hot(edge_attr.long(), num_classes=5).float()
        g.edata['ue_mask'] = torch.ones(g.num_edges()).bool()

        return cls(g, atom_type_map=atom_type_map)

    # this code is adapted from MiDi: https://github.com/cvignac/MiDi/blob/ba07fc5b1313855c047ba0b90e7aceae47e34e38/midi/analysis/rdkit_functions.py
    def build_molecule(self):
        mol = build_molecule(self.positions, self.atom_types, self.atom_charges, self.bond_src_idxs, self.bond_dst_idxs, self.bond_types)
        return mol

    def compute_valencies(self):
        """Compute the valencies of every atom in the molecule. Returns a tensor of shape (num_atoms,)."""
        adj = torch.zeros((self.num_atoms, self.num_atoms))
        adjusted_bond_types = self.bond_types.clone()
        adjusted_bond_types[adjusted_bond_types == 4] = 1.5
        adjusted_bond_types = adjusted_bond_types.float()
        adj[self.bond_src_idxs, self.bond_dst_idxs] = adjusted_bond_types
        adj[self.bond_dst_idxs, self.bond_src_idxs] = adjusted_bond_types
        valencies = torch.sum(adj, dim=-1).long()
        return valencies

    def process_traj_frames(self, traj_frames: Dict[str, torch.Tensor], ep_traj: bool = False):
        """Converts the trajectory frames to a list of rdkit molecules."""
        # convert the frames to a list of rdkit molecules
        g_dummy = copy_graph(self.g)

        if ep_traj:
            n_frames = traj_frames['x_1_pred'].shape[0]
            x_final = traj_frames['x_1_pred'][-1]
        else:
            n_frames = traj_frames['x'].shape[0]
            x_final = traj_frames['x'][-1] # has shape (n_atoms, 3)

        traj_mols = []
        for frame_idx in range(n_frames):

            # put current frame data into graph
            for feat in traj_frames.keys():

                if '1_pred' in feat:
                    continue

                if feat == 'e':
                    data_src = g_dummy.edata
                else:
                    data_src = g_dummy.ndata

                if ep_traj:
                    traj_key = f'{feat}_1_pred'
                else:
                    traj_key = feat

                data_src[f'{feat}_1'] = traj_frames[traj_key][frame_idx].clone()

            # extract mol data from graph
            positions, atom_types, atom_charges, bond_types, bond_src_idxs, bond_dst_idxs = extract_moldata_from_graph(
                g_dummy,
                self.atom_type_map,
                ctmc_mol=self.ctmc_mol)

            # align positions to final frame
            if self.align_traj:
                positions = rigid_alignment(positions, x_final)

            # build rdkit molecule
            mol = build_molecule(positions, atom_types, atom_charges, bond_src_idxs, bond_dst_idxs, bond_types)

            # add mol to list
            traj_mols.append(mol)

        traj_mols = [mol for mol in traj_mols if mol is not None]

        if len(traj_mols) < len(traj_frames):
            print(f'WARNING: {len(traj_frames) - len(traj_mols)} frames were not converted to rdkit molecules')

        return traj_mols


def extract_moldata_from_graph(g: dgl.DGLGraph, atom_type_map: List[str], exclude_charges: bool = False, ctmc_mol: bool = False):

    # extract node-level features
    positions = g.ndata['x_1']

    # extract node-level features
    positions = g.ndata['x_1']
    atom_types = g.ndata['a_1'].argmax(dim=1)
    atom_types = [atom_type_map[int(atom)] for atom in atom_types]

    if exclude_charges:
        atom_charges = None
    else:
        atom_charges = g.ndata['c_1'].argmax(dim=1) - 2 # implicit assumption that index 0 charge is -2


    # get bond types and atom indicies for every edge, convert types from simplex to integer
    bond_types = g.edata['e_1'].argmax(dim=1)
    bond_types[bond_types == 5] = 0 # set masked bonds to 0
    bond_src_idxs, bond_dst_idxs = g.edges()

    # get just the upper triangle of the adjacency matrix
    upper_edge_mask = g.edata['ue_mask']
    bond_types = bond_types[upper_edge_mask]
    bond_src_idxs = bond_src_idxs[upper_edge_mask]
    bond_dst_idxs = bond_dst_idxs[upper_edge_mask]

    # get only non-zero bond types
    bond_mask = bond_types != 0
    bond_types = bond_types[bond_mask]
    bond_src_idxs = bond_src_idxs[bond_mask]
    bond_dst_idxs = bond_dst_idxs[bond_mask]


    return positions, atom_types, atom_charges, bond_types, bond_src_idxs, bond_dst_idxs


def build_molecule(positions, atom_types, atom_charges, bond_src_idxs, bond_dst_idxs, bond_types):
    """Builds a rdkit molecule from the given atom and bond information."""
    # create a rdkit molecule and add atoms to it
    mol = Chem.RWMol()
    for atom_type, charge in zip(atom_types, atom_charges):
        a = Chem.Atom(atom_type)
        if charge != 0:
            a.SetFormalCharge(int(charge))
        mol.AddAtom(a)

    # add bonds to rdkit molecule
    for bond_type, src_idx, dst_idx in zip(bond_types, bond_src_idxs, bond_dst_idxs):
        src_idx = int(src_idx)
        dst_idx = int(dst_idx)
        mol.AddBond(src_idx, dst_idx, bond_type_map[bond_type])

    try:
        mol = mol.GetMol()
    except Chem.KekulizeException:
        return None

    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = positions[i]
        x, y, z = float(x), float(y), float(z)
        conf.SetAtomPosition(i, Point3D(x,y,z))
    mol.AddConformer(conf)

    return mol


def copy_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:

    # get edges
    edges = g.edges(form='uv')

    # get number of nodes
    num_nodes = g.num_nodes()

    g_copy = dgl.graph(edges, num_nodes=num_nodes, device=g.device)

    # transfer over node features
    for nfeat in g.ndata.keys():
        g_copy.ndata[nfeat] = g.ndata[nfeat].detach().clone()

    # transfer over edge features
    for efeat in g.edata.keys():
        g_copy.edata[efeat] = g.edata[efeat].detach().clone()


    return g_copy

def dataset_mol_to_sampled_mol(g, atom_type_map) -> SampledMolecule:
    for feat in 'xace':
        if feat == 'e':
            data_src = g.edata
        else:
            data_src = g.ndata
        data_src[f'{feat}_1'] = data_src[f'{feat}_1_true']

    g.edata['ue_mask'] = get_upper_edge_mask(g)
    return SampledMolecule(g, atom_type_map)

def dataset_mol_to_rdmol(g, atom_type_map):
    dataset_mol_to_sampled_mol(g, atom_type_map).rdkit_mol

# ========================================================================================
# Sampling analysis helper (non-trainable)
# ========================================================================================


allowed_bonds = {
    'H': {0: 1, 1: 0, -1: 0},
    'C': {0: [3, 4], 1: 3, -1: 3},
    'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},
    'O': {0: 2, 1: 3, -1: 1},
    'F': {0: 1, -1: 0},
    'B': 3, 'Al': 3, 'Si': 4,
    'P': {0: [3, 5], 1: 4},
    'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    'Cl': 1, 'As': 3,
    'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5],
    'Se': [2, 4, 6],
}


def check_stability(molecule: SampledMolecule):
    atom_types = molecule.atom_types
    valencies = molecule.valencies
    atom_charges = molecule.atom_charges
    if atom_charges is None:
        atom_charges = torch.zeros(len(atom_types), dtype=torch.long)

    n_stable_atoms = 0
    mol_stable = True
    for atom_type, valency, charge in zip(atom_types, valencies, atom_charges):
        valency = int(valency)
        charge = int(charge)
        possible_bonds = allowed_bonds.get(atom_type)
        if possible_bonds is None:
            is_stable = False
        elif isinstance(possible_bonds, int):
            is_stable = possible_bonds == valency
        elif isinstance(possible_bonds, dict):
            expected = possible_bonds.get(charge, possible_bonds.get(0))
            is_stable = expected == valency if isinstance(expected, int) else valency in expected
        else:
            is_stable = valency in possible_bonds
        mol_stable = mol_stable and bool(is_stable)
        n_stable_atoms += int(is_stable)
    return n_stable_atoms, mol_stable


class SampleAnalyzer:
    """Original validity/stability interface used by FlowMol and sampling scripts."""

    def __init__(self, processed_data_dir: str | None = None, dataset: str = 'geom'):
        self.processed_data_dir = processed_data_dir
        self.dataset = dataset

    def compute_validity(self, sampled_molecules: List[SampledMolecule], return_counts: bool = False):
        n_valid = 0
        num_components = []
        frag_fracs = []
        for sampled in sampled_molecules:
            mol = sampled.rdkit_mol
            if mol is None:
                continue
            try:
                frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                num_components.append(len(frags))
                largest = max(frags, default=mol, key=lambda item: item.GetNumAtoms())
                frag_fracs.append(largest.GetNumAtoms() / sampled.num_atoms)
                Chem.SanitizeMol(largest)
                Chem.MolToSmiles(largest)
                n_valid += 1
            except Exception:
                continue

        n = len(sampled_molecules)
        frac_valid = n_valid / n if n else 0.0
        avg_frag = float(np.mean(frag_fracs)) if frag_fracs else 0.0
        avg_components = float(np.mean(num_components)) if num_components else 0.0
        if return_counts:
            return (frac_valid, avg_frag, avg_components, n_valid,
                    sum(frag_fracs), len(frag_fracs), sum(num_components), len(num_components))
        return frac_valid, avg_frag, avg_components

    def analyze(self, sampled_molecules: List[SampledMolecule], return_counts: bool = False,
                energy_div: bool = False, functional_validity: bool = False):
        n_atoms = 0
        n_stable_atoms = 0
        n_stable_molecules = 0
        for molecule in sampled_molecules:
            n_atoms += molecule.num_atoms
            stable_atoms, stable_molecule = check_stability(molecule)
            n_stable_atoms += stable_atoms
            n_stable_molecules += int(stable_molecule)

        validity = self.compute_validity(sampled_molecules, return_counts=return_counts)
        if return_counts:
            frac_valid, avg_frag, avg_components, n_valid, sum_frag, n_frag, sum_comp, n_comp = validity
        else:
            frac_valid, avg_frag, avg_components = validity

        result = {
            'frac_atoms_stable': n_stable_atoms / n_atoms if n_atoms else 0.0,
            'frac_mols_stable_valence': n_stable_molecules / len(sampled_molecules) if sampled_molecules else 0.0,
            'frac_valid_mols': frac_valid,
            'avg_frag_frac': avg_frag,
            'avg_num_components': avg_components,
        }
        if return_counts:
            return {
                'n_stable_atoms': n_stable_atoms,
                'n_atoms': n_atoms,
                'n_stable_molecules': n_stable_molecules,
                'n_molecules': len(sampled_molecules),
                'n_valid': n_valid,
                'sum_frag_fracs': sum_frag,
                'n_frag_fracs': n_frag,
                'sum_num_components': sum_comp,
                'n_num_components': n_comp,
            }
        return result

# ========================================================================================
# Equivariant vector fields (original molguidance/models/vector_field.py)
# ========================================================================================

class EndpointVectorField(nn.Module):

    def __init__(self, n_atom_types: int,
                    canonical_feat_order: list,
                    interpolant_scheduler: InterpolantScheduler,
                    n_charges: int = 6,
                    n_bond_types: int = 5,
                    n_vec_channels: int = 16,
                    n_cp_feats: int = 0,
                    n_hidden_scalars: int = 64,
                    n_hidden_edge_feats: int = 64,
                    n_recycles: int = 1,
                    n_molecule_updates: int = 2,
                    convs_per_update: int = 2,
                    n_message_gvps: int = 3,
                    n_update_gvps: int = 3,
                    separate_mol_updaters: bool = False,
                    message_norm: Union[float, str] = 100,
                    update_edge_w_distance: bool = False,
                    rbf_dmax = 20,
                    rbf_dim = 16,
                    exclude_charges: bool = False,
                    continuous_inv_temp_schedule = None,
                    continuous_inv_temp_max: float = 10.0,
                    has_mask: bool = False # if we are using CTMC, input categorical features will have mask tokens,
                    # this means their one-hot representations will have an extra dimension,
                    # and the neural network instantiated by this method need to account for this
                    # it is definitely anti-pattern to have a parameter in parent class that is only needed for one sub-class (CTMCVectorField)
                    # however, this is the fastest way to get CTMCVectorField working right now, so we will be anti-pattern for the sake of time
    ):
        super().__init__()

        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.n_hidden_scalars = n_hidden_scalars
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.message_norm = message_norm
        self.n_recycles = n_recycles
        self.separate_mol_updaters: bool = separate_mol_updaters
        self.exclude_charges = exclude_charges
        self.interpolant_scheduler = interpolant_scheduler
        self.canonical_feat_order = canonical_feat_order

        if self.exclude_charges:
            self.n_charges = 0

        self.convs_per_update = convs_per_update
        self.n_molecule_updates = n_molecule_updates

        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        assert n_vec_channels >= 3, 'n_vec_channels must be >= 3'

        self.continuous_inv_temp_schedule = continuous_inv_temp_schedule
        self.continouts_inv_temp_max = continuous_inv_temp_max
        self.continuous_inv_temp_func = self.build_continuous_inv_temp_func(self.continuous_inv_temp_schedule, self.continouts_inv_temp_max)

        self.n_cat_feats = { # number of possible values for each categorical variable (not including mask tokens in the case of CTMC)
            'a': n_atom_types,
            'c': n_charges,
            'e': n_bond_types
        }

        n_mask_feats = int(has_mask)
        self.n_mask_feats = n_mask_feats

        node_input_dim = (
            n_atom_types
            + (0 if exclude_charges else n_charges)
            + 1
            + n_mask_feats * (1 + int(not exclude_charges))
        )
        self.scalar_embedding = nn.Sequential(
            nn.Linear(node_input_dim, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_scalars)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(n_bond_types + n_mask_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_edge_feats)
        )

        self.conv_layers = []
        for conv_idx in range(convs_per_update*n_molecule_updates):
            self.conv_layers.append(GVPConv(
                scalar_size=n_hidden_scalars,
                vector_size=n_vec_channels,
                n_cp_feats=n_cp_feats,
                edge_feat_size=n_hidden_edge_feats,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                message_norm=message_norm,
                rbf_dmax=rbf_dmax,
                rbf_dim=rbf_dim
            )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        # create molecule update layers
        self.node_position_updaters = nn.ModuleList([])
        self.edge_updaters = nn.ModuleList([])
        if self.separate_mol_updaters:
            n_updaters = n_molecule_updates
        else:
            n_updaters = 1
        for _ in range(n_updaters):
            self.node_position_updaters.append(NodePositionUpdate(n_hidden_scalars, n_vec_channels, n_gvps=3, n_cp_feats=n_cp_feats))
            self.edge_updaters.append(EdgeUpdate(n_hidden_scalars, n_hidden_edge_feats, update_edge_w_distance=update_edge_w_distance, rbf_dim=rbf_dim))


        self.node_output_head = nn.Sequential(
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_atom_types + n_charges)
        )

        self.to_edge_logits = nn.Sequential(
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_bond_types)
        )

    def build_continuous_inv_temp_func(self, schedule, max_inv_temp=None):

        if schedule is None:
            inv_temp_func = lambda t: 1.0
        elif schedule == 'linear':
            inv_temp_func = lambda t: max_inv_temp*(1 - t)
        elif callable(schedule):
            inv_temp_func = schedule
        else:
            raise ValueError(f'Invalid continuous_inv_temp_schedule: {schedule}')
        return inv_temp_func

    def molecule_updater_index(self, conv_idx: int) -> int | None:
        """Return the updater used after a convolution, or None.

        Molecular state is updated after every convs_per_update convolutions.
        Subtracting one converts the completed group count to a zero-based
        updater index, including the convs_per_update == 1 case.
        """
        if (conv_idx + 1) % self.convs_per_update != 0:
            return None
        if not self.separate_mol_updaters:
            return 0
        updater_idx = (conv_idx + 1) // self.convs_per_update - 1
        if updater_idx >= len(self.node_position_updaters):
            raise IndexError(
                f"Updater index {updater_idx} exceeds "
                f"{len(self.node_position_updaters)} configured updaters"
            )
        return updater_idx


    def forward(self, g: dgl.DGLGraph, t: torch.Tensor,
                 node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, apply_softmax=False, remove_com=False):
        """Predict x_1 (trajectory destination) given x_t"""
        device = g.device

        with g.local_scope():
            # gather node and edge features for input to convolutions
            node_scalar_features = [
                g.ndata['a_t'],
                t[node_batch_idx].unsqueeze(-1)
            ]

            # if we are not excluding charges, include them in the node scalar features
            if not self.exclude_charges:
                node_scalar_features.append(g.ndata['c_t'])

            node_scalar_features = torch.cat(node_scalar_features, dim=-1)
            node_scalar_features = self.scalar_embedding(node_scalar_features)

            node_positions = g.ndata['x_t']

            num_nodes = g.num_nodes()

            # initialize the vector features for every node to be zeros
            node_vec_features = torch.zeros((num_nodes, self.n_vec_channels, 3), device=device)
            # i thought setting the first three channels to the identity matrix would be a good idea,
            # but this actually breaks rotational equivariance
            # node_vec_features[:, :3, :] = torch.eye(3, device=device).unsqueeze(0).repeat(num_nodes, 1, 1)

            edge_features = g.edata['e_t']
            edge_features = self.edge_embedding(edge_features)

            x_diff, d = self.precompute_distances(g)
            for recycle_idx in range(self.n_recycles):
                for conv_idx, conv in enumerate(self.conv_layers):

                    # perform a single convolution which updates node scalar and vector features (but not positions)
                    node_scalar_features, node_vec_features = conv(g,
                            scalar_feats=node_scalar_features,
                            coord_feats=node_positions,
                            vec_feats=node_vec_features,
                            edge_feats=edge_features,
                            x_diff=x_diff,
                            d=d
                    )

                    # every convs_per_update convolutions, update the node positions and edge features
                    updater_idx = self.molecule_updater_index(conv_idx)
                    if updater_idx is not None:

                        node_positions = self.node_position_updaters[updater_idx](node_scalar_features, node_positions, node_vec_features)

                        x_diff, d = self.precompute_distances(g, node_positions)

                        edge_features = self.edge_updaters[updater_idx](g, node_scalar_features, edge_features, d=d)


            # predict final charges and atom type logits
            node_scalar_features = self.node_output_head(node_scalar_features)
            atom_type_logits = node_scalar_features[:, :self.n_atom_types]
            if not self.exclude_charges:
                atom_charge_logits = node_scalar_features[:, self.n_atom_types:]

            # predict the final edge logits
            ue_feats = edge_features[upper_edge_mask]
            le_feats = edge_features[~upper_edge_mask]
            edge_logits = self.to_edge_logits(ue_feats + le_feats)

            # project node positions back into zero-COM subspace
            if remove_com:
                g.ndata['x_1_pred'] = node_positions
                g.ndata['x_1_pred'] = g.ndata['x_1_pred'] - dgl.readout_nodes(g, feat='x_1_pred', op='mean')[node_batch_idx]
                node_positions = g.ndata['x_1_pred']

        # build a dictionary of predicted features
        dst_dict = {
            'x': node_positions,
            'a': atom_type_logits,
            'e': edge_logits
        }
        if not self.exclude_charges:
            dst_dict['c'] = atom_charge_logits

        # apply softmax to categorical features, if requested
        # at training time, we don't want to apply softmax because we use cross-entropy loss which includes softmax
        # at inference time, we want to apply softmax to get a vector which lies on the simplex
        if apply_softmax:
            for feat in dst_dict.keys():
                if feat in ['a', 'c', 'e']: # if this is a categorical feature
                    dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1) # apply softmax to this feature

        return dst_dict

    def precompute_distances(self, g: dgl.DGLGraph, node_positions=None):
        """Precompute the pairwise distances between all nodes in the graph."""

        with g.local_scope():

            if node_positions is None:
                g.ndata['x_d'] = g.ndata['x_t']
            else:
                g.ndata['x_d'] = node_positions

            g.apply_edges(dgl_fn.u_sub_v("x_d", "x_d", "x_diff"))
            dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
            x_diff = g.edata['x_diff'] / dij
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)

        return x_diff, d

    def integrate(self, g: dgl.DGLGraph,
        node_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor,
        n_timesteps: int,
        visualize=False, **kwargs):
        """Integrate the trajectories of molecules along the vector field."""

        # get the timepoint for integration
        t = torch.linspace(0, 1, n_timesteps, device=g.device)

        # get the corresponding alpha values for each timepoint
        alpha_t = self.interpolant_scheduler.alpha_t(t) # has shape (n_timepoints, n_feats)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # set x_t = x_0
        for feat in self.canonical_feat_order:
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_t'] = data_src[f'{feat}_0']


        # if visualizing the trajectory, create a datastructure to store the trajectory
        if visualize:
            traj_frames = {}
            for feat in self.canonical_feat_order:
                if feat == "e":
                    data_src = g.edata
                    split_sizes = g.batch_num_edges()
                else:
                    data_src = g.ndata
                    split_sizes = g.batch_num_nodes()

                split_sizes = split_sizes.detach().cpu().tolist()
                init_frame = data_src[f'{feat}_0'].detach().cpu()
                init_frame = torch.split(init_frame, split_sizes)
                traj_frames[feat] = [ init_frame ]
                traj_frames[f'{feat}_1_pred'] = []

        for s_idx in range(1,t.shape[0]):

            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            # compute next step and set x_t = x_s
            g = self.step(g, s_i, t_i, alpha_t_i, alpha_s_i, alpha_t_prime_i, node_batch_idx, upper_edge_mask, **kwargs)

            if visualize:
                for feat in self.canonical_feat_order:

                    if feat == "e":
                        g_data_src = g.edata
                    else:
                        g_data_src = g.ndata

                    if feat == 'e':
                        split_sizes = g.batch_num_edges()
                    else:
                        split_sizes = g.batch_num_nodes()
                    split_sizes = split_sizes.detach().cpu().tolist()
                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[feat].append(frame)


                    # record endpoint frame for visualization
                    ep_key = f'{feat}_1_pred'
                    if ep_key not in g_data_src:
                        # the endpoint key wont be there for VectorField because
                        # i haven't dervived a method of obtaining intermediate xhats from the vector field
                        continue
                    ep_frame = g_data_src[ep_key].detach().cpu()
                    ep_frame = torch.split(ep_frame, split_sizes)
                    traj_frames[ep_key].append(ep_frame)

        # set x_1 = x_t
        for feat in self.canonical_feat_order:

            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata

            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:

            # currently, traj_frames[key] is a list of lists. each sublist contains the frame for every molecule in the batch
            # we want to rearrange this so that traj_frames is a list of dictionaries, where each dictionary contains the frames for a single molecule
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in traj_frames.keys():
                    feat_traj = []
                    n_frames = len(traj_frames[feat])
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)


            return g, reshaped_traj_frames

        return g

    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor,
             inv_temp_func=None,
            **kwargs):

        if inv_temp_func is None:
            inv_temp_func = self.continuous_inv_temp_func

        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g,
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
        )

        # compute x_s for each feature and set x_t = x_s
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == "e":
                data_src = g.edata
            else:
                data_src = g.ndata

            x_t = data_src[f'{feat}_t']
            x_1 = dst_dict[feat]

            if feat == "e":
                x_t = x_t[upper_edge_mask]

            # evaluate the vector field at the current timepoint
            vf = self.vector_field(x_t, x_1, alpha_t_i[feat_idx], alpha_t_prime_i[feat_idx])

            # apply temperature scaling
            vf = vf*inv_temp_func(t_i)

            # x1_weight = alpha_t_prime_i[feat_idx]*(s_i - t_i)/(1 - alpha_t_i[feat_idx])
            # xt_weight = 1 - x1_weight

            # apply euler integration step
            x_s = x_t + vf*(s_i - t_i)

            if feat == "e":

                # set the edge features so that corresponding upper and lower triangle edges have the same value
                e_s = torch.zeros_like(g.edata['e_0'])
                e_s[upper_edge_mask] = x_s
                e_s[~upper_edge_mask] = x_s
                x_s = e_s

                e_1 = torch.zeros_like(g.edata['e_0'])
                e_1[upper_edge_mask] = dst_dict[feat]
                e_1[~upper_edge_mask] = dst_dict[feat]
                x_1 = e_1

            # record predicted endoint, for visualization purposes
            data_src[f'{feat}_1_pred'] = x_1.detach().clone()

            # record updated feature in the graph
            data_src[f'{feat}_t'] = x_s

        return g


    def vector_field(self, x_t, x_1, alpha_t, alpha_t_prime):
        vf = alpha_t_prime/(1 - alpha_t) * (x_1 - x_t)
        return vf


    def sample_conditional_path(self, g, t, node_batch_idx, edge_batch_idx, upper_edge_mask):
        """Interpolate between the prior and true terminal state of the ligand."""
        # upper_edge_mask is not used here but it is needed for DirichletVectorField and we need to keep consistent
        # function signatures across vector field classes so that MolFM can use them interchangeably
        src_weights, dst_weights = self.interpolant_scheduler.interpolant_weights(t)

        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if feat == 'e':
                continue

            src_weight, dst_weight = src_weights[:, feat_idx][node_batch_idx].unsqueeze(-1), dst_weights[:, feat_idx][node_batch_idx].unsqueeze(-1)
            g.ndata[f'{feat}_t'] = src_weight * g.ndata[f'{feat}_0'] + dst_weight * g.ndata[f'{feat}_1_true']

        e_idx = self.canonical_feat_order.index('e')
        src_weight, dst_weight = src_weights[:, e_idx][edge_batch_idx].unsqueeze(-1), dst_weights[:, e_idx][edge_batch_idx].unsqueeze(-1)
        g.edata[f'e_t'] = src_weight * g.edata[f'e_0'] + dst_weight * g.edata[f'e_1_true']

        return g


class VectorField(EndpointVectorField):

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor,
                 node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, apply_softmax=False, remove_com=False):

        dst_dict = super().forward(g, t, node_batch_idx, upper_edge_mask, apply_softmax, remove_com)
        dst_dict['x'] = dst_dict['x'] - g.ndata['x_t']
        return dst_dict

    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):

        # predict the destination of the trajectory given the current timepoint
        vec_field = self(
            g,
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=False,
            remove_com=False
        )

        # compute x_s for each feature and set x_t = x_s
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if feat == "e":
                x_t = g.edata[f'e_t'][upper_edge_mask]
            else:
                x_t = g.ndata[f'{feat}_t']

            # x_s = x_t + vec_field*(s - t)
            x_s = x_t + vec_field[feat]*(s_i - t_i)

            # set x_t = x_s
            if feat == "e":
                x_t = torch.zeros_like(g.edata['e_0'])
                x_t[upper_edge_mask] = x_s
                x_t[~upper_edge_mask] = x_s
                g.edata[f'{feat}_t'] = x_t
            else:
                x_t = x_s
                g.ndata[f'{feat}_t'] = x_t

        # remove COM from x_t
        g.ndata['x_t'] = g.ndata['x_t'] - dgl.readout_nodes(g, feat='x_t', op='mean')[node_batch_idx]

        return g


class DirichletVectorField(EndpointVectorField):

    def __init__(self, *args, w_max=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_max = w_max
        self.categorical_condflows = {}
        self.categorical_condflows['a'] = DirichletConditionalFlow(K=self.n_atom_types, alpha_min=0, alpha_max=w_max+2, alpha_spacing=0.01)
        self.categorical_condflows['c'] = DirichletConditionalFlow(K=self.n_charges, alpha_min=0, alpha_max=w_max+2, alpha_spacing=0.01)
        self.categorical_condflows['e'] = DirichletConditionalFlow(K=self.n_bond_types, alpha_min=0, alpha_max=w_max+2, alpha_spacing=0.01)


        self.n_cat_dict = {
            'a': self.n_atom_types,
            'c': self.n_charges,
            'e': self.n_bond_types
        }

    def alpha_to_w(self, alpha_t):
        return alpha_t*self.w_max + 1

    def sample_conditional_path(self, g, t, node_batch_idx, edge_batch_idx, upper_edge_mask):
        """Interpolate between the prior and true terminal state of the ligand."""
        # TODO: this computation could be made more efficient by concatenating node features and edge features into a single tensor and then interpolate them all at once before splitting them back up
        alpha_t = self.interpolant_scheduler.alpha_t(t) # has shape (n_timepoints, n_feats)
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            # skip the bond orders, its too clunky to incorpoarte them into this loop due to upper/lower triangle edge indexing
            if feat == 'e':
                continue

            if feat == 'x':
                src_weights, dst_weights = 1 - alpha_t, alpha_t
                src_weight, dst_weight = src_weights[:, feat_idx][node_batch_idx].unsqueeze(-1), dst_weights[:, feat_idx][node_batch_idx].unsqueeze(-1)
                g.ndata[f'{feat}_t'] = src_weight * g.ndata[f'{feat}_0'] + dst_weight * g.ndata[f'{feat}_1_true']
            else: # now we are doing categorical node features (a,c)
                alpha_expanded = alpha_t[:, feat_idx][node_batch_idx].unsqueeze(-1)
                w_t = self.alpha_to_w(alpha_expanded)
                dirichlet_params = torch.ones_like(g.ndata[f'{feat}_1_true']) + w_t*g.ndata[f'{feat}_1_true']
                g.ndata[f'{feat}_t'] = torch.distributions.Dirichlet(dirichlet_params).sample()

        # sample condiitonal path for edge features
        e_idx = self.canonical_feat_order.index('e')
        alpha_expanded = alpha_t[:, e_idx][edge_batch_idx][upper_edge_mask].unsqueeze(-1)
        w_t = self.alpha_to_w(alpha_expanded)
        dirichlet_params = torch.ones_like(g.edata[f'e_1_true'][upper_edge_mask]) + w_t*(g.edata[f'e_1_true'][upper_edge_mask])
        ue_samples = torch.distributions.Dirichlet(dirichlet_params).sample()
        g.edata['e_t'] = torch.zeros_like(g.edata['e_1_true'])
        g.edata[f'e_t'][upper_edge_mask] = ue_samples
        g.edata[f'e_t'][~upper_edge_mask] = ue_samples

        return g

    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):

        # alpha_t_i has shape (n_feats,)

        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g,
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True
        )

        # take integration step for positions
        x_1 = dst_dict['x']
        x_t = g.ndata['x_t']
        vf = self.vector_field(x_t, x_1, alpha_t_i[0], alpha_t_prime_i[0])
        g.ndata['x_t'] = x_t + (s_i - t_i)*vf

        # record predicted endoint, for visualization purposes
        g.ndata['x_1_pred'] = x_1.detach().clone()

        # convert alpha values to w
        w_t = self.alpha_to_w(alpha_t_i)
        w_s = self.alpha_to_w(alpha_s_i)


        # take integration step for node categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat not in ['a', 'c']:
                continue
            w_t_feat = w_t[feat_idx]
            w_s_feat = w_s[feat_idx]
            x_t = g.ndata[f'{feat}_t'] # has shape (n_nodes, n_cat)

            c_factor = self.categorical_condflows[feat].c_factor(
                x_t.cpu().numpy(),
                w_t_feat.item()
            )
            # c_factor has shape equal to x_t, which is (n_nodes, n_cat)
            c_factor = torch.from_numpy(c_factor).to(g.device).float()
            if torch.isnan(c_factor).any():
                # print(f'NAN cfactor after: xt.min(): {xt.min()}, out_probs.min(): {out_probs.min()}')
                print('NAN c_factor')
                c_factor = torch.nan_to_num(c_factor)

            # get possible endpoints as one-hot vectors
            eps = torch.eye(self.n_cat_dict[feat], device=g.device) # shape (n_cat, n_cat)

            # compute conditional vector fields for each possible endpoint
            cond_vec_fields = (eps[:, None, :] - x_t[None, :, :]) * c_factor.unsqueeze(0) # has shape (n_cat, n_nodes, n_cat)
            endpoint_probs = dst_dict[feat] # has shape (n_nodes, n_cat)
            endpoint_probs = endpoint_probs.transpose(0, 1).unsqueeze(-1) # has shape (n_cat, n_nodes, 1)
            marginal_vec_field = ( endpoint_probs * cond_vec_fields ).sum(dim=0)

            # take integration step
            x_s = x_t + marginal_vec_field*(w_s_feat - w_t_feat)

            # project onto simplex if necessary
            x_s = self.project_simplex(x_s)

            # set x_t = x_s
            g.ndata[f'{feat}_t'] = x_s

            # record predicted endoint, for visualization purposes
            g.ndata[f'{feat}_1_pred'] = dst_dict[feat].detach().clone()

        # now we compute marginal vector field and take a step for edge features
        e_idx = self.canonical_feat_order.index('e')
        w_t_e = w_t[e_idx]
        w_s_e = w_s[e_idx]
        x_t = g.edata['e_t'][upper_edge_mask] # has shape (n_edges, n_cat)
        c_factor = self.categorical_condflows['e'].c_factor(
            x_t.cpu().numpy(),
            w_t_e.item()
        )
        c_factor = torch.from_numpy(c_factor).to(g.device).float()
        if torch.isnan(c_factor).any():
            # print(f'NAN cfactor after: xt.min(): {xt.min()}, out_probs.min(): {out_probs.min()}')
            print('NAN c_factor')
            c_factor = torch.nan_to_num(c_factor)
        # get possible endpoints as one-hot vectors
        eps = torch.eye(self.n_cat_dict['e'], device=g.device) # shape (n_cat, n_cat)

        # compute conditional vector fields for each possible endpoint
        cond_vec_fields = (eps[:, None, :] - x_t[None, :, :]) * c_factor.unsqueeze(0) # has shape (n_cat, n_edges, n_cat)
        endpoint_probs = dst_dict[feat] # has shape (n_edges, n_cat)
        endpoint_probs = endpoint_probs.transpose(0, 1).unsqueeze(-1) # has shape (n_cat, n_edges, 1)
        marginal_vec_field = ( endpoint_probs * cond_vec_fields ).sum(dim=0)
        x_s = x_t + marginal_vec_field*(w_s_e - w_t_e)
        g.edata['e_t'][upper_edge_mask] = x_s
        g.edata['e_t'][~upper_edge_mask] = x_s

        # record predicted endpoint for bond orders
        e_1_pred = torch.zeros_like(g.edata['e_0'])
        e_1_pred[upper_edge_mask] = dst_dict['e']
        e_1_pred[~upper_edge_mask] = dst_dict['e']
        g.edata['e_1_pred'] = e_1_pred

        return g

    def project_simplex(self, x_s: torch.Tensor):
        n, c = x_s.shape
        ref_sum = torch.ones(n, dtype=x_s.dtype, device=x_s.device)
        if not torch.allclose(x_s.sum(dim=-1), ref_sum, atol=1e-4) or not (x_s >= 0).all():
            # print(f'WARNING: x_t.min(): {x_s.min()}. Some values of x_s do not lie on the simplex. There are {(x_s<0).sum()} negative values in x_s of shape {x_s.shape} that are negative.')
            x_s = simplex_proj(x_s)
        return x_s

class NodePositionUpdate(nn.Module):

    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0):
        super().__init__()

        self.gvps = []
        for i in range(n_gvps):

            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            self.gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation,
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)

class EdgeUpdate(nn.Module):

    def __init__(self, n_node_scalars, n_edge_feats, update_edge_w_distance=False, rbf_dim=16):
        super().__init__()

        self.update_edge_w_distance = update_edge_w_distance

        input_dim = n_node_scalars*2 + n_edge_feats
        if update_edge_w_distance:
            input_dim += rbf_dim

        self.edge_update_fn = nn.Sequential(
            nn.Linear(input_dim, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )

        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats, d):


        # get indicies of source and destination nodes
        src_idxs, dst_idxs = g.edges()

        mlp_inputs = [
            node_scalars[src_idxs],
            node_scalars[dst_idxs],
            edge_feats,
        ]

        if self.update_edge_w_distance:
            mlp_inputs.append(d)

        edge_feats = self.edge_norm(edge_feats + self.edge_update_fn(torch.cat(mlp_inputs, dim=-1)))
        return edge_feats

# ========================================================================================
# CTMC vector field (original molguidance/models/ctmc_vector_field.py)
# ========================================================================================

PROPERTY_MAP = {
        'A': 0, 'B': 1, 'C': 2, 'mu': 3, 'alpha': 4,
        'homo': 5, 'lumo': 6, 'gap': 7, 'r2': 8,
        'zpve': 9, 'u0': 10, 'u298': 11, 'h298': 12,
        'g298': 13, 'cv': 14, 'u0_atom': 15, 'u298_atom': 16,
        'h298_atom': 17, 'g298_atom': 18
    }

class CTMCVectorField(EndpointVectorField):

    # uses Continuous-Time Markov Chain (CTMC) to model the flow of cateogrical features (atom type, charge, bond order)
    # CTMC for flow-matching was originally proposed in https://arxiv.org/abs/2402.04997

    # we make some modifications to the original CTMC model:
    # our conditional trajectories interpolate along a progress coordiante alpha_t, which is a function of time t
    # where we set a different alpha_t for each data modality
    # we also do purity sampling in a slightly different way that in theory would be slightly less performant but is
    # computationally much more efficient when working with batched graphs

    def __init__(self, *args,
                 stochasticity: float = 0.0,
                 high_confidence_threshold: float = 0.0,
                 dfm_type: str = 'campbell',
                 cat_temperature_schedule: Union[str, Callable, float] = 0.05,
                 cat_temp_decay_max: float = 0.8,
                 cat_temp_decay_a: float = 2,
                 forward_weight_schedule: Union[str, Callable, float] = 'beta',
                 fw_beta_a: float = 0.25, fw_beta_b: float = 0.25, fw_beta_max: float = 10.0, property_embedding_dim: int = 64,
                 training_mode:bool=True,
                 conditional_generation:bool=True,
                 property_embedder=None,
                 properties_handle_method:str=None,
                 dataset_name:str = "qm9",
                 **kwargs):
        super().__init__(*args, has_mask=True, **kwargs) # initialize endpoint vector field
        self.property_embedding_dim = property_embedding_dim
        self.training_mode = training_mode
        self.conditional_generation = conditional_generation
        self.property_embedder = property_embedder
        self.properties_handle_method = properties_handle_method
        # Normalization metadata are immutable during a sampling run. Cache the
        # loaded PT payload instead of reading it once per integration timestep.
        self._normalization_cache_path = None
        self._normalization_cache = None

        self.eta = stochasticity # default stochasticity parameter, 0 means no stochasticity
        self.hc_thresh = high_confidence_threshold # the threshold for for calling a prediction high-confidence, 0 means no purity sampling
        self.dfm_type = dfm_type
        self.dataset_name = dataset_name

        # configure temperature schedule for categorical features
        self.cat_temperature_schedule = cat_temperature_schedule
        self.cat_temp_decay_max = cat_temp_decay_max
        self.cat_temp_decay_a = cat_temp_decay_a
        self.cat_temp_func = self.build_cat_temp_schedule(
            cat_temperature_schedule=cat_temperature_schedule,
            cat_temp_decay_max=cat_temp_decay_max,
            cat_temp_decay_a=cat_temp_decay_a)

        # configure forward weight schedule
        self.forward_weight_schedule = forward_weight_schedule
        self.fw_beta_a = fw_beta_a
        self.fw_beta_b = fw_beta_b
        self.fw_beta_max = fw_beta_max
        self.forward_weight_func = self.build_fw_schedule(
            forward_weight_schedule=forward_weight_schedule,
            fw_beta_a=fw_beta_a,
            fw_beta_b=fw_beta_b,
            fw_beta_max=fw_beta_max)

        if self.dfm_type not in ['campbell', 'gat', 'campbell_rate_matrix']:
            raise ValueError(f"Invalid dfm_type: {self.dfm_type}")

        self.mask_idxs = { # for each categorical feature, the index of the mask token
            'a': self.n_atom_types,
            'c': self.n_charges,
            'e': self.n_bond_types,
        }

        # Two separate embeddings for conditional and unconditional cases
        input_dim_base = (
            self.n_atom_types
            + self.n_charges
            + 1
            + self.n_mask_feats * (1 + int(not self.exclude_charges))
        )
        self.scalar_embedding_uncond = nn.Sequential(
            nn.Linear(input_dim_base, self.n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(self.n_hidden_scalars, self.n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(self.n_hidden_scalars)
        )

        self.scalar_embedding_cond = nn.Sequential(
            nn.Linear(self.n_hidden_scalars + property_embedding_dim, self.n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(self.n_hidden_scalars, self.n_hidden_scalars),
            # nn.SiLU(),
            # nn.LayerNorm(self.n_hidden_scalars)
        )

    def build_cat_temp_schedule(self, cat_temperature_schedule, cat_temp_decay_max, cat_temp_decay_a):

        if cat_temperature_schedule == 'decay':
            cat_temp_func = lambda t: cat_temp_decay_max*torch.pow(1-t, cat_temp_decay_a)
        elif isinstance(cat_temperature_schedule, (float, int)):
            cat_temp_func = lambda t: cat_temperature_schedule
        elif callable(cat_temperature_schedule):
            cat_temp_func = cat_temperature_schedule
        else:
            raise ValueError(f"Invalid cat_temperature_schedule: {cat_temperature_schedule}")

        return cat_temp_func

    def build_fw_schedule(self, forward_weight_schedule, fw_beta_a, fw_beta_b, fw_beta_max):

        if forward_weight_schedule == 'beta':
            forward_weight_func = lambda t: 1 + fw_beta_max*torch.pow(t, fw_beta_a)*torch.pow(1-t, fw_beta_b)
        elif isinstance(forward_weight_schedule, (float, int)):
            forward_weight_func = lambda t: forward_weight_schedule
        elif callable(forward_weight_schedule):
            forward_weight_func = forward_weight_schedule
        else:
            raise ValueError(f"Invalid forward_weight_schedule: {forward_weight_schedule}")

        return forward_weight_func

    def sample_conditional_path(self, g, t, node_batch_idx, edge_batch_idx, upper_edge_mask):
        # sample p(g_t|g_0,g_1)
        # this includes the standard probability path for positions and CTMC probability paths for categorical features
        # t has shape (batch_size,)
        _, alpha_t = self.interpolant_scheduler.interpolant_weights(t)
        batch_size = g.batch_size
        num_nodes = g.num_nodes()
        device = g.device

        # alpha_t has shape (batch_size, 4)

        # sample positions at time t
        x_idx = self.canonical_feat_order.index('x')
        dst_weight = alpha_t[:, x_idx][node_batch_idx].unsqueeze(-1)
        src_weight = 1 - dst_weight
        g.ndata['x_t'] = src_weight*g.ndata['x_0'] + dst_weight*g.ndata['x_1_true']

        # sample categorical node features
        t_node = t[node_batch_idx]
        for feat in ('a', 'c'):
            if feat not in self.canonical_feat_order:
                continue
            feat_idx = self.canonical_feat_order.index(feat)

            # all ground-truth categorical variables are set to one-hot representations without mask token by dataloader class
            # so here we convert to token indicies by argmaxing, and then one-hot encode again but with mask token

            # set x_t = x_1 to start
            xt = g.ndata[f'{feat}_1_true'].argmax(-1) # has shape (num_nodes,)
            alpha_t_feat = alpha_t[:, feat_idx][node_batch_idx] # has shape (num_nodes,)

            # set each node's feature to the mask token with probability 1 - alpha_t
            xt[ torch.rand(num_nodes, device=device) < 1 - alpha_t_feat ] = self.mask_idxs[feat]
            g.ndata[f'{feat}_t'] = one_hot(xt, num_classes=self.n_cat_feats[feat]+1)

        # sample categorical edge features
        num_edges = g.num_edges() / 2
        num_edges = int(num_edges)
        edge_feat_idx = self.canonical_feat_order.index('e')
        alpha_t_e = alpha_t[:, edge_feat_idx][edge_batch_idx][upper_edge_mask]
        et_upper = g.edata['e_1_true'][upper_edge_mask].argmax(-1)
        et_upper[ torch.rand(num_edges, device=device) < 1 - alpha_t_e ] = self.mask_idxs['e']

        n,d = g.edata['e_1_true'].shape
        e_t = torch.zeros((n,d+1), dtype=g.edata['e_1_true'].dtype, device=g.device)
        et_upper_onehot = one_hot(et_upper, num_classes=self.n_cat_feats['e']+1).float()
        e_t[upper_edge_mask] = et_upper_onehot
        e_t[~upper_edge_mask] = et_upper_onehot
        g.edata['e_t'] = e_t

        return g

    def integrate(self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor, n_timesteps: int,
        visualize=False,
        dfm_type='campbell',
        stochasticity=8.0,
        high_confidence_threshold=0.9,
        cat_temp_func=None,
        forward_weight_func=None,
        tspan=None,
        normalization_file_path:str=None,
        conditional_generation:bool=True,
        property_name:str=None,
        properties_for_sampling:int|float=None,
        training_mode:bool=True,
        properties_handle_method:str=None,
        multilple_values_to_one_property: List[float|int] | None = None,
        **kwargs):
        """Integrate the trajectories of molecules along the vector field."""

        # TODO: this overrides EndpointVectorField.integrate just because it has some extra arguments
        # we should refactor this so that we don't have to copy the entire function

        self.properties_for_sampling = properties_for_sampling
        self.property_name = property_name
        self.conditional_generation = conditional_generation
        self.normalization_file_path = normalization_file_path
        self.training_mode = training_mode
        self.properties_handle_method = properties_handle_method
        self.multilple_values_to_one_property = multilple_values_to_one_property

        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func
        if forward_weight_func is None:
            forward_weight_func = self.forward_weight_func

        # get edge_batch_idx
        edge_batch_idx = get_edge_batch_idxs(g)

        # get the timepoint for integration
        if tspan is None:
            t = torch.linspace(0, 1, n_timesteps, device=g.device)
        else:
            t = tspan

        # get the corresponding alpha values for each timepoint
        alpha_t = self.interpolant_scheduler.alpha_t(t) # has shape (n_timepoints, n_feats)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # set x_t = x_0
        for feat in self.canonical_feat_order:
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_t'] = data_src[f'{feat}_0']


        # if visualizing the trajectory, create a datastructure to store the trajectory
        if visualize:
            traj_frames = {}
            for feat in self.canonical_feat_order:
                if feat == "e":
                    data_src = g.edata
                    split_sizes = g.batch_num_edges()
                else:
                    data_src = g.ndata
                    split_sizes = g.batch_num_nodes()

                split_sizes = split_sizes.detach().cpu().tolist()
                init_frame = data_src[f'{feat}_0'].detach().cpu()
                init_frame = torch.split(init_frame, split_sizes)
                traj_frames[feat] = [ init_frame ]
                traj_frames[f'{feat}_1_pred'] = []

        for s_idx in range(1,t.shape[0]):

            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            # determine if this is the last integration step
            if s_idx == t.shape[0] - 1:
                last_step = True
            else:
                last_step = False

            # compute next step and set x_t = x_s
            g = self.step(g, s_i, t_i, alpha_t_i, alpha_s_i,
                alpha_t_prime_i,
                node_batch_idx,
                edge_batch_idx,
                upper_edge_mask,
                cat_temp_func=cat_temp_func,
                forward_weight_func=forward_weight_func,
                dfm_type=dfm_type,
                stochasticity=stochasticity,
                high_confidence_threshold=high_confidence_threshold,
                last_step=last_step,
                normalization_file_path=normalization_file_path,
                conditional_generation=conditional_generation,
                property_name=property_name,
                properties_for_sampling=properties_for_sampling,
                training_mode=training_mode,
                **kwargs)

            if visualize:
                for feat in self.canonical_feat_order:

                    if feat == "e":
                        g_data_src = g.edata
                    else:
                        g_data_src = g.ndata

                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    if feat == 'e':
                        split_sizes = g.batch_num_edges()
                    else:
                        split_sizes = g.batch_num_nodes()
                    split_sizes = split_sizes.detach().cpu().tolist()
                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[feat].append(frame)

                    ep_frame = g_data_src[f'{feat}_1_pred'].detach().cpu()
                    ep_frame = torch.split(ep_frame, split_sizes)
                    traj_frames[f'{feat}_1_pred'].append(ep_frame)

        # set x_1 = x_t
        for feat in self.canonical_feat_order:

            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata

            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:

            # currently, traj_frames[key] is a list of lists. each sublist contains the frame for every molecule in the batch
            # we want to rearrange this so that traj_frames is a list of dictionaries, where each dictionary contains the frames for a single molecule
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in traj_frames.keys():
                    feat_traj = []
                    n_frames = len(traj_frames[feat])
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)


            return g, reshaped_traj_frames

        return g

    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, edge_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor,
             cat_temp_func: Callable,
             forward_weight_func: Callable,
             dfm_type: str = 'campbell',
             stochasticity: float = 8.0,
             high_confidence_threshold: float = 0.9,
             last_step: bool = False,
             inv_temp_func: Callable = None,
            normalization_file_path:str=None,
            conditional_generation:bool=True,
            property_name:str=None,
            properties_for_sampling:int|float=None,
            training_mode:bool=True,
            ):

        device = g.device

        if stochasticity is None:
            eta = self.eta
        else:
            eta = stochasticity

        if high_confidence_threshold is None:
            hc_thresh = self.hc_thresh
        else:
            hc_thresh = high_confidence_threshold

        if dfm_type is None:
            dfm_type = self.dfm_type

        if inv_temp_func is None:
            inv_temp_func = lambda t: 1.0

        if conditional_generation and not training_mode:
            assert self.properties_for_sampling is not None or self.multilple_values_to_one_property is not None , "Properties for sampling must be provided for conditional generation"

        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g,
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
        )

        dt = s_i - t_i

        # take integration step for positions
        x_1 = dst_dict['x']
        x_t = g.ndata['x_t']
        vf = self.vector_field(x_t, x_1, alpha_t_i[0], alpha_t_prime_i[0])
        g.ndata['x_t'] = x_t + dt*vf*inv_temp_func(t_i)

        # record predicted endpoint for visualization
        g.ndata['x_1_pred'] = x_1.detach().clone()

        # take integration step for node categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'x':
                continue

            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            xt = data_src[f'{feat}_t'].argmax(-1) # has shape (num_nodes,)

            if feat == 'e':
                xt = xt[upper_edge_mask]

            p_s_1 = dst_dict[feat]
            temperature = cat_temp_func(t_i)
            p_s_1 = F.softmax(torch.log(p_s_1)/temperature, dim=-1) # log probabilities

            if dfm_type == 'campbell':


                xt, x_1_sampled = \
                self.campbell_step(p_1_given_t=p_s_1,
                                xt=xt,
                                stochasticity=eta,
                                hc_thresh=hc_thresh,
                                alpha_t=alpha_t_i[feat_idx],
                                alpha_t_prime=alpha_t_prime_i[feat_idx],
                                dt=dt,
                                batch_size=g.batch_size,
                                batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(),
                                n_classes=self.n_cat_feats[feat]+1,
                                mask_index=self.mask_idxs[feat],
                                last_step=last_step,
                                batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx,
                                )

            elif dfm_type == 'gat':
                # record predicted endpoint for visualization
                x_1_sampled = torch.cat([p_s_1, torch.zeros_like(p_s_1[:, :1])], dim=-1)

                xt = self.gat_step(
                    p_1_given_t=p_s_1,
                    xt=xt,
                    alpha_t=alpha_t_i[feat_idx],
                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                    forward_weight=forward_weight_func(t_i),
                    dt=dt,
                    batch_size=g.batch_size,
                    batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(),
                    n_classes=self.n_cat_feats[feat]+1,
                    mask_index=self.mask_idxs[feat],
                    batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx,
                )


            # if we are doing edge features, we need to modify xt and x_1_sampled to have upper and lower edges
            if feat == 'e':
                e_t = torch.zeros_like(g.edata['e_t'])
                e_t[upper_edge_mask] = xt
                e_t[~upper_edge_mask] = xt
                xt = e_t

                e_1_sampled = torch.zeros_like(g.edata['e_t'])
                e_1_sampled[upper_edge_mask] = x_1_sampled
                e_1_sampled[~upper_edge_mask] = x_1_sampled
                x_1_sampled = e_1_sampled

            data_src[f'{feat}_t'] = xt
            data_src[f'{feat}_1_pred'] = x_1_sampled

        return g


    def campbell_step(self, p_1_given_t: torch.Tensor,
                      xt: torch.Tensor,
                      stochasticity: float,
                      hc_thresh: float,
                      alpha_t: float,
                      alpha_t_prime: float,
                      dt,
                      batch_size: int,
                      batch_num_nodes: torch.Tensor,
                      n_classes: int,
                      mask_index:int,
                      last_step: bool,
                      batch_idx: torch.Tensor,
    ):
        if not bool(torch.isfinite(p_1_given_t).all()):
            bad_rows = ~torch.isfinite(p_1_given_t).all(dim=-1)
            raise FloatingPointError(
                "Non-finite endpoint probabilities passed to campbell_step: "
                f"{int(bad_rows.sum())}/{p_1_given_t.shape[0]} invalid rows"
            )
        row_sums = p_1_given_t.sum(dim=-1, keepdim=True)
        if bool((row_sums <= 0).any()):
            raise FloatingPointError(
                "Zero-sum endpoint probabilities passed to campbell_step"
            )
        # Categorical validates the simplex quite strictly. Renormalize after
        # temperature scaling to remove harmless floating-point sum drift.
        p_1_given_t = p_1_given_t.clamp_min(0.0) / row_sums
        x1 = Categorical(p_1_given_t).sample() # has shape (num_nodes,)

        unmask_prob = dt*( alpha_t_prime + stochasticity*alpha_t  ) / (1 - alpha_t)
        mask_prob = dt*stochasticity

        unmask_prob = torch.clamp(unmask_prob, min=0, max=1)
        mask_prob = torch.clamp(mask_prob, min=0, max=1)

        # sample which nodes will be unmasked
        if hc_thresh > 0:
            # select more high-confidence predictions for unmasking than low-confidence predictions
            will_unmask = purity_sampling(
                xt=xt, x1=x1, x1_probs=p_1_given_t, unmask_prob=unmask_prob,
                mask_index=mask_index, batch_size=batch_size, batch_num_nodes=batch_num_nodes,
                node_batch_idx=batch_idx, hc_thresh=hc_thresh, device=xt.device)
        else:
            # uniformly sample nodes to unmask
            will_unmask = torch.rand(xt.shape[0], device=xt.device) < unmask_prob
            will_unmask = will_unmask * (xt == mask_index) # only unmask nodes that are currently masked

        if not last_step:
            # compute which nodes will be masked
            will_mask = torch.rand(xt.shape[0], device=xt.device) < mask_prob
            will_mask = will_mask * (xt != mask_index) # only mask nodes that are currently unmasked

            # mask the nodes
            xt[will_mask] = mask_index

        # unmask the nodes
        xt[will_unmask] = x1[will_unmask]

        xt = one_hot(xt, num_classes=n_classes).float()
        x1 = one_hot(x1, num_classes=n_classes).float()
        return xt, x1

    def gat_step(self,
                p_1_given_t: torch.Tensor,
                xt: torch.Tensor,
                alpha_t: float,
                alpha_t_prime: float,
                forward_weight: float,
                dt,
                batch_size: int,
                batch_num_nodes: torch.Tensor,
                n_classes: int,
                mask_index:int,
                batch_idx: torch.Tensor,
):


        # add a zero-column on to p_1_given_t to represent the mask token
        p_1_given_t = torch.cat([p_1_given_t, torch.zeros_like(p_1_given_t[:, :1])], dim=-1)

        # create a one-hot encoding of xt
        delta_xt = one_hot(xt, num_classes=n_classes).float()

        # compute forward probability velocity
        u_forward = alpha_t_prime / (1 - alpha_t) * (p_1_given_t - delta_xt)

        # create a delta on the mask token
        delta_mask = torch.zeros_like(delta_xt)
        delta_mask[:, mask_index] = 1

        # compute the backward probability velocity
        u_backward = alpha_t_prime / (alpha_t + 1e-8) * (delta_xt - delta_mask)

        # compute the probability velocity
        backward_weight = forward_weight - 1
        pvel = forward_weight*u_forward - backward_weight*u_backward

        # compute the parameters of the transition distritibution
        p_step = delta_xt + dt*pvel

        # clamp p_step to be valid
        p_step = torch.clamp(p_step, min=1.0e-9, max=1)

        # sample x_{t+dt} from the transition distribution
        x_dt = Categorical(p_step).sample()

        # one-hot encode x_{t+dt}
        x_dt = one_hot(x_dt, num_classes=n_classes).float()

        return x_dt

####################### conditional
    def _load_normalization_params(self, path: str | Path | None):
        if path is None:
            return None
        resolved = str(Path(path).expanduser().resolve())
        if self._normalization_cache_path != resolved:
            payload = torch.load(resolved, map_location='cpu', weights_only=False)
            if 'mean' not in payload or 'std' not in payload:
                raise KeyError('Normalization file must contain mean and std')
            self._normalization_cache_path = resolved
            self._normalization_cache = payload
        return self._normalization_cache

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor,
                node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor,
                apply_softmax=False, remove_com=False):
        device = g.device

        with g.local_scope():
            # Determine if this is conditional generation
            is_conditional = len(t.shape) > 1 or self.conditional_generation

            # Gather base features
            time_tensor = t[:, 0] if len(t.shape) > 1 else t

            base_features = [
                g.ndata['a_t'],
                time_tensor[node_batch_idx].unsqueeze(-1)
            ]
            if not self.exclude_charges:
                base_features.append(g.ndata['c_t'])
            base_features = torch.cat(base_features, dim=-1)
            node_scalar_features = self.scalar_embedding_uncond(base_features)

            try:
                if is_conditional:
                    # Initialize prop_emb as None
                    prop_emb = None

                    # Case 1: Property info in t (for training)
                    if len(t.shape) > 1:
                        prop_emb = t[:, 1:][node_batch_idx]

                    # Case 2: Explicit sampling properties (for sampling)
                    elif self.properties_for_sampling is not None or self.multilple_values_to_one_property is not None:
                        # Convert scalar to tensor properly
                        if self.properties_for_sampling is not None:
                            assert isinstance(self.properties_for_sampling, (int, float))

                        # Load normalization parameters if needed
                        norm_params = self._load_normalization_params(
                            self.normalization_file_path
                        )
                        if norm_params is not None:
                            mean_value = norm_params['mean']
                            std_value = norm_params['std']
                            if torch.is_tensor(mean_value) and mean_value.ndim > 0:
                                if self.property_name in PROPERTY_MAP and mean_value.numel() > 1:
                                    property_idx = int(PROPERTY_MAP[self.property_name])
                                    mean_value = mean_value[property_idx]
                                    std_value = std_value[property_idx]
                                elif mean_value.numel() == 1:
                                    mean_value = mean_value.reshape(())
                                    std_value = std_value.reshape(())
                                else:
                                    raise ValueError(
                                        'A vector normalization file requires a known property_name; '
                                        f'got {self.property_name!r}'
                                    )
                            mean = float(mean_value.item() if torch.is_tensor(mean_value) else mean_value)
                            std = float(std_value.item() if torch.is_tensor(std_value) else std_value)
                            if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
                                raise ValueError(
                                    f'Invalid normalization values: mean={mean}, std={std}'
                                )
                            target_transform = str(norm_params.get('target_transform', 'identity')).lower()
                            target_epsilon = float(norm_params.get('target_epsilon', 0.0))

                        def _transform_raw_property(value):
                            if norm_params is None:
                                return value
                            if target_transform in ('identity', 'none', 'raw', ''):
                                return value
                            if target_transform in ('log', 'ln', 'natural_log'):
                                return math.log(float(value) + target_epsilon)
                            if target_transform in ('log10', 'log_10'):
                                return math.log10(float(value) + target_epsilon)
                            if target_transform in ('log1p', 'ln1p'):
                                return math.log1p(float(value) + target_epsilon)
                            raise ValueError(f'Unsupported target transform: {target_transform}')

                        if self.multilple_values_to_one_property:
                            assert isinstance(self.multilple_values_to_one_property, list)
                            if norm_params is not None:
                                properties_list = [(_transform_raw_property(val) - mean) / std for val in self.multilple_values_to_one_property]
                                properties_batch = torch.tensor(properties_list, device=device).view(g.batch_size, 1)
                            else:
                                properties_batch = torch.tensor(self.multilple_values_to_one_property, device=device).view(g.batch_size, 1)
                        else:
                            properties_for_sampling = self.properties_for_sampling
                            if norm_params is not None:
                                properties_for_sampling = (_transform_raw_property(properties_for_sampling) - mean) / std
                            properties_batch = torch.full((g.batch_size, 1), properties_for_sampling, device=device)

                        # Get embedding
                        prop_emb = self.property_embedder(properties_batch)

                        # Repeat for each node in graph
                        prop_emb = prop_emb[node_batch_idx]

                    if prop_emb is None:
                        raise ValueError("No property information available for conditional generation")

                    # Handle properties with different methods
                    if self.properties_handle_method == 'concatenate_sum':
                        intermediate_features = torch.cat([node_scalar_features, prop_emb], dim=-1)
                        intermediate_features = self.scalar_embedding_cond(intermediate_features)
                        node_scalar_features = node_scalar_features + intermediate_features
                    elif self.properties_handle_method == 'concatenate':
                        intermediate_features = torch.cat([node_scalar_features, prop_emb], dim=-1)
                        node_scalar_features = self.scalar_embedding_cond(intermediate_features)
                    elif self.properties_handle_method == 'sum':
                        node_scalar_features = node_scalar_features + prop_emb
                    elif self.properties_handle_method == 'multiply':
                        prop_emb = torch.sigmoid(prop_emb) + 0.5 # range (0.5, 1.5)
                        node_scalar_features = node_scalar_features * prop_emb
                    elif self.properties_handle_method == 'concatenate_multiply':
                        intermediate_features = torch.cat([node_scalar_features, prop_emb], dim=-1)
                        intermediate_features = self.scalar_embedding_cond(intermediate_features)
                        intermediate_features = torch.sigmoid(intermediate_features) + 0.5
                        node_scalar_features = node_scalar_features * intermediate_features
                    else:
                        raise ValueError(f"Invalid properties_handle_method: {self.properties_handle_method}")

            except Exception as e:
                print(f"Debug info: is_conditional={is_conditional}, "
                    # f"training_mode={self.training_mode}, "
                    f"t.shape={t.shape}, "
                    f"has_prop={hasattr(g, 'prop')}, "
                    # f"properties_for_sampling={self.properties_for_sampling}"
                    )
                raise e

            # Rest of forward remains same as EndpointVectorField
            node_positions = g.ndata['x_t']
            num_nodes = g.num_nodes()
            node_vec_features = torch.zeros((num_nodes, self.n_vec_channels, 3), device=device)
            edge_features = g.edata['e_t']
            edge_features = self.edge_embedding(edge_features)

            x_diff, d = self.precompute_distances(g)
            for recycle_idx in range(self.n_recycles):
                for conv_idx, conv in enumerate(self.conv_layers):

                    # perform a single convolution which updates node scalar and vector features (but not positions)
                    node_scalar_features, node_vec_features = conv(g,
                            scalar_feats=node_scalar_features,
                            coord_feats=node_positions,
                            vec_feats=node_vec_features,
                            edge_feats=edge_features,
                            x_diff=x_diff,
                            d=d
                    )

                    # every convs_per_update convolutions, update the node positions and edge features
                    updater_idx = self.molecule_updater_index(conv_idx)
                    if updater_idx is not None:

                        node_positions = self.node_position_updaters[updater_idx](node_scalar_features, node_positions, node_vec_features)

                        x_diff, d = self.precompute_distances(g, node_positions)

                        edge_features = self.edge_updaters[updater_idx](g, node_scalar_features, edge_features, d=d)


            # predict final charges and atom type logits
            node_scalar_features = self.node_output_head(node_scalar_features)
            atom_type_logits = node_scalar_features[:, :self.n_atom_types]
            if not self.exclude_charges:
                atom_charge_logits = node_scalar_features[:, self.n_atom_types:]

            # predict the final edge logits
            ue_feats = edge_features[upper_edge_mask]
            le_feats = edge_features[~upper_edge_mask]
            edge_logits = self.to_edge_logits(ue_feats + le_feats)

            # project node positions back into zero-COM subspace
            if remove_com:
                g.ndata['x_1_pred'] = node_positions
                g.ndata['x_1_pred'] = g.ndata['x_1_pred'] - dgl.readout_nodes(g, feat='x_1_pred', op='mean')[node_batch_idx]
                node_positions = g.ndata['x_1_pred']

        # build a dictionary of predicted features
        dst_dict = {
            'x': node_positions,
            'a': atom_type_logits,
            'e': edge_logits
        }
        if not self.exclude_charges:
            dst_dict['c'] = atom_charge_logits

        # apply softmax to categorical features, if requested
        # at training time, we don't want to apply softmax because we use cross-entropy loss which includes softmax
        # at inference time, we want to apply softmax to get a vector which lies on the simplex
        if apply_softmax:
            for feat in dst_dict.keys():
                if feat in ['a', 'c', 'e']: # if this is a categorical feature
                    dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1) # apply softmax to this feature

        return dst_dict

# ========================================================================================
# FlowMol Lightning model (original molguidance/models/flowmol.py)
# ========================================================================================

class FlowMol(pl.LightningModule):

    canonical_feat_order = ['x', 'a', 'c', 'e']
    node_feats = ['x', 'a', 'c']
    edge_feats = ['e']

    def __init__(self,
                 atom_type_map: List[str],
                 n_atoms_hist_file: str,
                 marginal_dists_file: str,
                 n_atom_charges: int = 6,
                 n_bond_types: int = 5,
                 sample_interval: float = 1.0, # how often to sample molecules from the model, measured in epochs
                 n_mols_to_sample: int = 64, # how many molecules to sample from the model during each sample/eval step during training
                 time_scaled_loss: bool = True,
                 exclude_charges: bool = False,
                 weight_ae: bool = False, # whether or not to apply weights to the atom and edge losses (infrequent categories given more weight)
                 target_blur: float = 0.0, # how much to blur the target distribution for categorical features
                 parameterization: str = 'ctmc', # how to parameterize the flow-matching objective, can be 'endpoint', 'vector-field', 'dirichlet' or 'ctmc'
                 total_loss_weights: Dict[str, float] | None = None,
                 lr_scheduler_config: dict | None = None,
                 interpolant_scheduler_config: dict | None = None,
                 vector_field_config: dict | None = None,
                 prior_config: dict | None = None,
                 default_n_timesteps: int = 250,
                 property_embedding_dim: int = 256,
                 conditional_generation: bool = True,
                 dataset_name: str = 'qm9',
                 gaussian_expansion: bool = False,
                 gaussian_start: float|None = None,
                 gaussian_stop: float|None = None,
                 n_gaussians: int = 5,
                 properties_handle_method: str = 'concatenate_sum', # choices ['concatenate', 'sum', 'multiply', 'concatenate_sum', 'concatenate_multiply']
                 conditioning_property: str | None = None,
                 property_transform: str | None = None,
                 property_epsilon: float = 1.0e-8,
                 property_normalization_file: str | None = None,
                 validation_seed: int = 12345,
                 compare_shuffled_condition: bool = False,
                 valence_loss_weight: float = 0.0,
                 charge_offset: int = 2,
                 log_intuitive_metrics: bool = True,
                 ):
        super().__init__()

        self.lr_scheduler_config = copy.deepcopy(lr_scheduler_config or {})
        interpolant_scheduler_config = copy.deepcopy(interpolant_scheduler_config or {})
        vector_field_config = copy.deepcopy(vector_field_config or {})
        self.prior_config = copy.deepcopy(prior_config or {})
        self.total_loss_weights = dict(total_loss_weights or {})
        self.canonical_feat_order = list(type(self).canonical_feat_order)
        self.node_feats = list(type(self).node_feats)
        self.edge_feats = list(type(self).edge_feats)
        self.atom_type_map = list(atom_type_map)
        self.n_atom_types = len(self.atom_type_map)
        self.n_atom_charges = n_atom_charges
        self.n_bond_types = n_bond_types
        self.time_scaled_loss = time_scaled_loss
        self.exclude_charges = exclude_charges
        self.marginal_dists_file = marginal_dists_file
        self.parameterization = parameterization
        self.weight_ae = weight_ae
        self.target_blur = target_blur
        self.n_atoms_hist_file = n_atoms_hist_file
        self.default_n_timesteps = default_n_timesteps
        self.conditional_generation = conditional_generation
        self.properties_handle_method = properties_handle_method
        self.property_embedding_dim = property_embedding_dim
        self.dataset_name = dataset_name
        self.conditioning_property = conditioning_property
        self.property_transform = property_transform
        self.property_epsilon = float(property_epsilon)
        self.property_normalization_file = property_normalization_file
        self.validation_seed = int(validation_seed)
        self.compare_shuffled_condition = bool(compare_shuffled_condition)
        self.valence_loss_weight = float(valence_loss_weight)
        self.charge_offset = int(charge_offset)
        self.log_intuitive_metrics = bool(log_intuitive_metrics)
        if self.valence_loss_weight < 0.0:
            raise ValueError("valence_loss_weight must be non-negative")

        valence_caps = []
        for symbol in self.atom_type_map:
            per_charge = []
            for charge_index in range(self.n_atom_charges):
                charge = charge_index - self.charge_offset
                if symbol == 'H':
                    cap = 1.0
                elif symbol == 'B':
                    cap = 4.0 if charge < 0 else 3.0
                elif symbol == 'C':
                    cap = 4.0
                elif symbol == 'N':
                    cap = 4.0 if charge > 0 else 3.0
                elif symbol == 'O':
                    cap = 3.0 if charge > 0 else 1.0 if charge < 0 else 2.0
                elif symbol in {'F', 'Cl', 'Br'}:
                    cap = 1.0
                elif symbol == 'S':
                    cap = 6.0
                else:
                    cap = 8.0
                per_charge.append(cap)
            valence_caps.append(per_charge)
        self.register_buffer(
            '_valence_caps',
            torch.tensor(valence_caps, dtype=torch.float32),
            persistent=False,
        )

        # for conditional generation of molecules with a property
        self.property_embedder = nn.Sequential(
            nn.Linear(1, property_embedding_dim),
            nn.SiLU(),
            nn.Linear(property_embedding_dim, property_embedding_dim),
            nn.LayerNorm(property_embedding_dim)
        )

        # remember set normalization to false in GaussianExpansion, otherwise start and stop values are wrong
        if gaussian_expansion:
            self.property_embedder = PropertyEmbedder(input_dim=1, embedding_dim=property_embedding_dim,
                                                         start=gaussian_start, stop=gaussian_stop,
                                                         n_gaussians=n_gaussians, use_activation=True)

        if self.weight_ae and parameterization == 'vector-field':
            raise NotImplementedError('weighting the atom and edge losses is not yet implemented for the vector-field parameterization')

        if self.target_blur != 0.0 and parameterization in ('vector-field', 'ctmc'):
            raise NotImplementedError(
                'target_blur is not supported for vector-field or CTMC parameterization'
            )

        if self.target_blur < 0.0:
            raise ValueError('target_blur must be non-negative')

        # if provided filepath to data dir does not exist, assume it is relative to the repo root
        processed_data_dir = Path(self.marginal_dists_file).parent
        if not processed_data_dir.exists():
            repo_root = Path(__file__).parent.parent.parent
            self.marginal_dists_file = repo_root / self.marginal_dists_file
            self.n_atoms_hist_file = repo_root / self.n_atoms_hist_file

        # do some boring stuff regarding the prior distribution
        self.configure_prior()

        if self.exclude_charges:
            self.node_feats.remove('c')
            self.canonical_feat_order.remove('c')
            self.total_loss_weights.pop('c', None)

        # create a dictionary mapping feature -> number of categories
        self.n_cat_dict = {
            'a': self.n_atom_types,
            'c': n_atom_charges,
            'e': n_bond_types,
        }

        for feat in self.canonical_feat_order:
            if feat not in self.total_loss_weights:
                self.total_loss_weights[feat] = 1.0

                # print warning if the user has not specified a loss weight for a feature
                print(f'WARNING: no loss weight specified for feature {feat}, using default of 1.0')

        self.exp_dist = Exponential(1.0)

        # construct histogram of number of atoms in each ligand
        self.build_n_atoms_dist(n_atoms_hist_file=self.n_atoms_hist_file)

        # create interpolant scheduler and vector field
        self.interpolant_scheduler = InterpolantScheduler(canonical_feat_order=self.canonical_feat_order,
                                                          **interpolant_scheduler_config)

        # check that a valid parameterization was specified
        if self.parameterization not in ['endpoint', 'vector-field', 'dirichlet', 'ctmc']:
            raise ValueError(f'parameterization must be one of "endpoint", "vector-field", or "dirichlet", "ctmc", got {self.parameterization}')

        if self.parameterization == 'endpoint':
            vector_field_class = EndpointVectorField
        elif self.parameterization == 'vector-field':
            vector_field_class = VectorField
        elif self.parameterization == 'dirichlet':
            vector_field_class = DirichletVectorField
        elif self.parameterization == 'ctmc':
            vector_field_class = CTMCVectorField

        if self.parameterization == 'ctmc':
            self.vector_field = vector_field_class(n_atom_types=self.n_atom_types,
                                            canonical_feat_order=self.canonical_feat_order,
                                            interpolant_scheduler=self.interpolant_scheduler,
                                            n_charges=n_atom_charges,
                                            n_bond_types=n_bond_types,
                                            exclude_charges=self.exclude_charges,
                                            property_embedding_dim=property_embedding_dim,
                                            property_embedder=self.property_embedder,
                                            properties_handle_method=self.properties_handle_method,
                                            conditional_generation=self.conditional_generation,
                                            dataset_name=dataset_name,
                                            **vector_field_config)
        else:
            self.vector_field = vector_field_class(n_atom_types=self.n_atom_types,
                                            canonical_feat_order=self.canonical_feat_order,
                                            interpolant_scheduler=self.interpolant_scheduler,
                                            n_charges=n_atom_charges,
                                            n_bond_types=n_bond_types,
                                            exclude_charges=self.exclude_charges,
                                            **vector_field_config)

        # Loss functions are created lazily on the active device. The charge
        # entry is removed there when exclude_charges=True.

        self.sample_interval = sample_interval # how often to sample molecules from the model, measured in epochs
        self.n_mols_to_sample = n_mols_to_sample # how many molecules to sample from the model during each sample/eval step during training
        self.last_sample_marker = 0 # this is the epoch_exact value of the last time we sampled molecules from the model
        self.sample_analyzer = SampleAnalyzer()


        # record the last epoch value for training steps -  this is really hacky but it lets me
        # align the validation losses with the correspoding training epoch value on W&B
        self.last_epoch_exact = 0

        self.save_hyperparameters()

    def configure_prior(self):
        # load the marginal distributions of atom types, bond orders and the conditional distribution of charges given atom type
        p_a, p_c, p_e, p_c_given_a = torch.load(
            self.marginal_dists_file, map_location='cpu', weights_only=False
        )
        self.p_a = p_a
        self.p_e = p_e

        # add the marginal distributions as arguments to the prior sampling functions
        if self.prior_config['a']['type'] == 'marginal':
            self.prior_config['a']['kwargs']['p'] = p_a

        if self.prior_config['e']['type'] == 'marginal':
            self.prior_config['e']['kwargs']['p'] = p_e

        if self.prior_config['c']['type'] == 'marginal':
            self.prior_config['c']['kwargs']['p'] = p_c

        if self.prior_config['c']['type'] == 'c-given-a':
            self.prior_config['c']['kwargs']['p_c_given_a'] = p_c_given_a

        if self.parameterization == 'dirichlet':
            for feat in ['a', 'c', 'e']:
                if self.prior_config[feat]['type'] != 'uniform-simplex':
                    raise ValueError('dirichlet parameterization requires that all categorical priors be uniform-simplex')

        if self.parameterization == 'ctmc':
            for feat in ['a', 'c', 'e']:
                if self.prior_config[feat]['type'] != 'ctmc':
                    raise ValueError('ctmc parameterization requires that all categorical priors be ctmc')

    def configure_loss_fns(self, device):
        # instantiate loss functions
        if self.time_scaled_loss:
            reduction = 'none'
        else:
            reduction = 'mean'

        if self.parameterization in  ['endpoint', 'dirichlet', 'ctmc']:
            categorical_loss_fn = nn.CrossEntropyLoss
        elif self.parameterization == 'vector-field':
            categorical_loss_fn = nn.MSELoss


        if self.weight_ae:
            a_kwargs = {'weight': (1 - self.p_a).to(device)}
            e_kwargs = {'weight': (1 - self.p_e).to(device)}
        else:
            a_kwargs = {}
            e_kwargs = {}

        if self.parameterization == 'ctmc':
            cat_kwargs = {'ignore_index': -100}
        else:
            cat_kwargs = {}

        self.loss_fn_dict = {
            'x': nn.MSELoss(reduction=reduction),
            'a': categorical_loss_fn(reduction=reduction, **a_kwargs, **cat_kwargs),
            'c': categorical_loss_fn(reduction=reduction, **cat_kwargs),
            'e': categorical_loss_fn(reduction=reduction, **e_kwargs, **cat_kwargs),
        }
        if self.exclude_charges:
            self.loss_fn_dict.pop('c', None)

    def compute_batch_losses(self, g: dgl.DGLGraph, stage: str) -> Dict[str, torch.Tensor]:
        """Compute feature losses for one batch.

        Subclasses can choose a different conditioning policy for training and
        validation.  The base model simply calls ``forward``.
        """
        return self(g)

    def compute_shuffled_condition_losses(
        self,
        g: dgl.DGLGraph,
    ) -> Dict[str, torch.Tensor] | None:
        return None

    def _combine_feature_losses(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), device=next(iter(losses.values())).device)
        for feat in self.canonical_feat_order:
            total = total + self.total_loss_weights[feat] * losses[feat]
        if 'valence' in losses:
            total = total + self.valence_loss_weight * losses['valence']
        if 'condition_margin' in losses:
            total = total + self.condition_margin_weight * losses['condition_margin']
        return total

    def _expected_valence_loss(
        self,
        g: dgl.DGLGraph,
        edge_logits: torch.Tensor,
        upper_edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize expected endpoint valence above the true atom/charge cap.

        This is differentiable with respect to bond logits and does not alter a
        sampled molecule. Ground-truth atom and charge classes define the cap so
        the model cannot reduce the penalty by predicting a more permissive atom.
        """
        state = self._valence_diagnostic_state(g, edge_logits, upper_edge_mask)
        overflow = state['expected_overflow']
        # Complete graphs contain O(N^2) candidate edges, so raw squared overflow
        # can dominate early training and bias the model toward predicting no bonds.
        # log1p is quadratic near zero but limits that early large-error influence.
        return torch.log1p(overflow.square()).mean()

    def _valence_diagnostic_state(
        self,
        g: dgl.DGLGraph,
        edge_logits: torch.Tensor,
        upper_edge_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if edge_logits.shape[-1] != 5:
            raise ValueError(
                "Valence loss expects bond classes "
                "[none, single, double, triple, aromatic]"
            )
        bond_orders = edge_logits.new_tensor([0.0, 1.0, 2.0, 3.0, 1.5])
        probabilities = torch.softmax(edge_logits, dim=-1)
        expected_orders = (probabilities * bond_orders).sum(-1)
        selected_orders = bond_orders[probabilities.argmax(dim=-1)]
        edge_src, edge_dst = g.edges()
        edge_src = edge_src[upper_edge_mask]
        edge_dst = edge_dst[upper_edge_mask]
        expected_valence = edge_logits.new_zeros(g.num_nodes())
        expected_valence.index_add_(0, edge_src, expected_orders)
        expected_valence.index_add_(0, edge_dst, expected_orders)
        selected_valence = edge_logits.new_zeros(g.num_nodes())
        selected_valence.index_add_(0, edge_src, selected_orders)
        selected_valence.index_add_(0, edge_dst, selected_orders)

        atom_indices = g.ndata['a_1_true'].argmax(dim=-1)
        if self.exclude_charges:
            charge_indices = torch.full_like(atom_indices, self.charge_offset)
        else:
            charge_indices = g.ndata['c_1_true'].argmax(dim=-1)
        caps = self._valence_caps[atom_indices, charge_indices].to(edge_logits.dtype)
        return {
            'expected_overflow': torch.relu(expected_valence - caps),
            'selected_overflow': torch.relu(selected_valence - caps),
        }

    @torch.no_grad()
    def _build_intuitive_validation_metrics(
        self,
        g: dgl.DGLGraph,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        upper_edge_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Translate denoising losses into human-readable validation metrics."""
        metrics: Dict[str, torch.Tensor] = {}
        coordinate_error = predictions['x'] - g.ndata['x_1_true']
        metrics['val_x_axis_rmse'] = coordinate_error.square().mean().sqrt()
        metrics['val_x_atom_rms_displacement'] = (
            coordinate_error.square().sum(dim=-1).mean().sqrt()
        )
        edge_src, edge_dst = g.edges()
        edge_src = edge_src[upper_edge_mask]
        edge_dst = edge_dst[upper_edge_mask]
        true_bond_mask = g.edata['e_1_true'][upper_edge_mask].argmax(dim=-1) != 0
        if bool(true_bond_mask.any()):
            bonded_src = edge_src[true_bond_mask]
            bonded_dst = edge_dst[true_bond_mask]
            predicted_lengths = torch.linalg.vector_norm(
                predictions['x'][bonded_src] - predictions['x'][bonded_dst],
                dim=-1,
            )
            true_lengths = torch.linalg.vector_norm(
                g.ndata['x_1_true'][bonded_src] - g.ndata['x_1_true'][bonded_dst],
                dim=-1,
            )
            metrics['val_x_true_bond_length_mae'] = (
                predicted_lengths - true_lengths
            ).abs().mean()

        for feat in ('a', 'c', 'e'):
            if feat not in predictions or feat not in targets:
                continue
            target = targets[feat]
            mask = target != -100
            if not bool(mask.any()):
                continue
            logits = predictions[feat][mask]
            labels = target[mask]
            nll = F.cross_entropy(logits, labels, reduction='mean')
            metrics[f'val_{feat}_masked_accuracy'] = (
                logits.argmax(dim=-1) == labels
            ).float().mean()
            metrics[f'val_{feat}_masked_perplexity'] = nll.exp()
            metrics[f'val_{feat}_masked_true_probability'] = (-nll).exp()
            if feat == 'e':
                metrics.update(self._bond_classification_metrics(logits, labels))

        valence = self._valence_diagnostic_state(
            g,
            predictions['e'],
            upper_edge_mask,
        )
        node_batch_idx = get_node_batch_idxs(g)
        for mode in ('expected', 'selected'):
            overflow = valence[f'{mode}_overflow']
            violating = overflow > 1.0e-6
            violating_per_molecule = overflow.new_zeros(g.batch_size)
            violating_per_molecule.index_add_(
                0,
                node_batch_idx,
                violating.to(overflow.dtype),
            )
            metrics[f'val_{mode}_valence_overflow_mean'] = overflow.mean()
            metrics[f'val_{mode}_valence_overflow_p95'] = torch.quantile(
                overflow,
                0.95,
            )
            metrics[f'val_{mode}_atom_valence_violation_rate'] = (
                violating.float().mean()
            )
            metrics[f'val_{mode}_molecule_valence_pass_rate'] = (
                violating_per_molecule == 0
            ).float().mean()
        return metrics

    def _bond_classification_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return bonded-edge and per-class diagnostics for masked edges."""
        predictions = logits.argmax(dim=-1)
        class_names = ('none', 'single', 'double', 'triple', 'aromatic')
        if logits.shape[-1] != len(class_names):
            raise ValueError(
                "Bond diagnostics expect classes "
                "[none, single, double, triple, aromatic]"
            )

        metrics: Dict[str, torch.Tensor] = {}
        eps = logits.new_tensor(1.0e-12)
        predicted_bond = predictions.ne(0)
        true_bond = labels.ne(0)
        bond_tp = (predicted_bond & true_bond).sum().to(logits.dtype)
        bond_precision = bond_tp / predicted_bond.sum().to(logits.dtype).clamp_min(eps)
        bond_recall = bond_tp / true_bond.sum().to(logits.dtype).clamp_min(eps)
        metrics['val_e_bonded_precision'] = bond_precision
        metrics['val_e_bonded_recall'] = bond_recall
        metrics['val_e_bonded_f1'] = (
            2.0 * bond_precision * bond_recall
            / (bond_precision + bond_recall).clamp_min(eps)
        )

        class_f1 = []
        for class_idx, class_name in enumerate(class_names):
            predicted_class = predictions.eq(class_idx)
            true_class = labels.eq(class_idx)
            true_positive = (predicted_class & true_class).sum().to(logits.dtype)
            precision = (
                true_positive
                / predicted_class.sum().to(logits.dtype).clamp_min(eps)
            )
            recall = (
                true_positive
                / true_class.sum().to(logits.dtype).clamp_min(eps)
            )
            f1 = (
                2.0 * precision * recall
                / (precision + recall).clamp_min(eps)
            )
            metrics[f'val_e_{class_name}_precision'] = precision
            metrics[f'val_e_{class_name}_recall'] = recall
            metrics[f'val_e_{class_name}_f1'] = f1
            class_f1.append(f1)
        metrics['val_e_macro_f1'] = torch.stack(class_f1).mean()
        return metrics

    @staticmethod
    def _raise_on_non_finite(
        losses: Dict[str, torch.Tensor],
        *,
        stage: str,
        batch_idx: int,
    ) -> None:
        non_finite = {
            key: value.detach()
            for key, value in losses.items()
            if not bool(torch.isfinite(value).all())
        }
        if non_finite:
            details = ', '.join(
                f"{key}={value.cpu().tolist()}"
                for key, value in non_finite.items()
            )
            raise FloatingPointError(
                f"Non-finite {stage} component loss at batch_idx={batch_idx}: {details}"
            )

    def training_step(self, g: dgl.DGLGraph, batch_idx: int):
        if not hasattr(self, 'batches_per_epoch'):
            self.batches_per_epoch = len(self.trainer.train_dataloader)

        epoch_exact = self.current_epoch + batch_idx / self.batches_per_epoch
        self.last_epoch_exact = epoch_exact
        self.lr_scheduler.step_lr(epoch_exact)

        losses = self.compute_batch_losses(g, stage='train')
        self._raise_on_non_finite(losses, stage='training', batch_idx=batch_idx)
        total_loss = self._combine_feature_losses(losses)

        self.log_dict(
            {
                f'train_{feat}_loss': value
                for feat, value in losses.items()
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=g.batch_size,
            sync_dist=True,
        )
        self.log(
            'train_total_loss',
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=g.batch_size,
            sync_dist=True,
        )
        return total_loss

    @staticmethod
    def _local_graph_copy(g: dgl.DGLGraph) -> dgl.DGLGraph:
        local = g.local_var()
        if hasattr(g, 'prop') and g.prop is not None:
            local.prop = g.prop.clone()
        return local

    def _validation_rng(self, g: dgl.DGLGraph, batch_idx: int):
        devices = []
        if g.device.type == 'cuda':
            devices = [g.device.index if g.device.index is not None else torch.cuda.current_device()]
        seed = self.validation_seed + int(self.global_rank) * 1_000_000 + int(batch_idx)
        return torch.random.fork_rng(devices=devices), seed

    def validation_step(self, g: dgl.DGLGraph, batch_idx: int):
        # Use the same corruption/time sample for a given validation batch at
        # every epoch. This reduces Monte-Carlo noise in checkpointing and early stopping.
        rng_context, seed = self._validation_rng(g, batch_idx)
        with rng_context:
            torch.manual_seed(seed)
            if g.device.type == 'cuda':
                torch.cuda.manual_seed(seed)
            losses = self.compute_batch_losses(
                self._local_graph_copy(g),
                stage='val',
            )

        self._raise_on_non_finite(losses, stage='validation', batch_idx=batch_idx)
        total_loss = self._combine_feature_losses(losses)

        val_logs = {
            f'val_cond_{feat}_loss': value
            for feat, value in losses.items()
        }
        val_logs['val_cond_total_loss'] = total_loss
        if self.log_intuitive_metrics:
            val_logs.update(getattr(self, '_last_intuitive_validation_metrics', {}))
            safe_total = total_loss.detach().abs().clamp_min(1.0e-12)
            for feat in self.canonical_feat_order:
                contribution = self.total_loss_weights[feat] * losses[feat].detach()
                val_logs[f'val_{feat}_loss_contribution_percent'] = (
                    100.0 * contribution / safe_total
                )
            if 'valence' in losses:
                contribution = self.valence_loss_weight * losses['valence'].detach()
                val_logs['val_valence_loss_contribution_percent'] = (
                    100.0 * contribution / safe_total
                )
        self.log_dict(
            val_logs,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=g.batch_size,
            sync_dist=True,
        )
        self.log(
            'val_cond_total_loss_bar',
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=g.batch_size,
            sync_dist=True,
        )
        if self.log_intuitive_metrics:
            intuitive = getattr(self, '_last_intuitive_validation_metrics', {})
            progress_metrics = {
                'val_atom_rms_bar': intuitive.get('val_x_atom_rms_displacement'),
                'val_bond_acc_bar': intuitive.get('val_e_masked_accuracy'),
                'val_valence_pass_bar': intuitive.get(
                    'val_selected_molecule_valence_pass_rate'
                ),
            }
            self.log_dict(
                {
                    name: value
                    for name, value in progress_metrics.items()
                    if value is not None
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=g.batch_size,
                sync_dist=True,
            )

        if self.compare_shuffled_condition and g.batch_size > 1:
            rng_context, seed = self._validation_rng(g, batch_idx)
            with rng_context:
                torch.manual_seed(seed)
                if g.device.type == 'cuda':
                    torch.cuda.manual_seed(seed)
                shuffled_losses = self.compute_shuffled_condition_losses(g)
            if shuffled_losses is not None:
                self._raise_on_non_finite(
                    shuffled_losses,
                    stage='shuffled-condition validation',
                    batch_idx=batch_idx,
                )
                shuffled_total = self._combine_feature_losses(shuffled_losses)
                self.log(
                    'val_shuffled_condition_total_loss',
                    shuffled_total,
                    on_step=False,
                    on_epoch=True,
                    batch_size=g.batch_size,
                    sync_dist=True,
                )
                self.log(
                    'val_condition_gain',
                    shuffled_total - total_loss,
                    on_step=False,
                    on_epoch=True,
                    batch_size=g.batch_size,
                    sync_dist=True,
                )
                self.log(
                    'val_condition_gain_percent',
                    100.0
                    * (shuffled_total - total_loss)
                    / shuffled_total.detach().abs().clamp_min(1.0e-12),
                    on_step=False,
                    on_epoch=True,
                    batch_size=g.batch_size,
                    sync_dist=True,
                )

        return total_loss

    def forward(self, g: dgl.DGLGraph):

        batch_size = g.batch_size
        device = g.device

        # check if the attribute loss_fn_dict exists
        # it is necessary to do this here (as opposed to in __init__) beacause
        # to instantiate the loss function class-conditioned weights, the weight
        # tensors need to be on the same device as the graph...seems pretty dumb but that's how it is
        if not hasattr(self, 'loss_fn_dict'):
            self.configure_loss_fns(device=g.device)

        # get batch indicies of every atom and edge
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # create a mask which selects all of the upper triangle edges from the batched graph
        upper_edge_mask = get_upper_edge_mask(g)

        # get initial COM of each molecule and remove the COM from the atom positions
        # this step is now done in MoleculeDataset.__getitem__ method, a necessary adjustment to do OT alignment on
        # the prior during the __getitem__ method
        # init_coms = dgl.readout_nodes(g, feat='x_1_true', op='mean')
        # g.ndata['x_1_true'] = g.ndata['x_1_true'] - init_coms[node_batch_idx]

        # sample molecules from prior
        # we used to sample the prior in the forward pass at training time,
        # but now at training time we sample the prior in the __getitem__ method of MoleculeDataset
        # this is so that we can compute OT alignments in parallel (since they cannot be done in batch)
        # g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

        # sample timepoints for each molecule in the batch
        t = torch.rand(batch_size, device=device).float()

        # Get property embeddings if available
        if hasattr(g, 'prop') and self.conditional_generation:
            self.property_embedder = self.property_embedder.to(device)
            g.prop = g.prop.unsqueeze(-1)
            g.prop = g.prop.to(device)
            prop_emb = self.property_embedder(g.prop)
            # Concatenate with time
            t_combined = torch.cat([t.unsqueeze(-1), prop_emb], dim=-1)
            # print("prop_emb\n", prop_emb.shape)
        else:
            t_combined = t
            print("no feature is used\n")

        # construct interpolated molecules
        g = self.vector_field.sample_conditional_path(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)

        # forward pass for the vector field
        vf_output = self.vector_field(g, t_combined, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)

        # get the target (label) for each feature
        targets = {}
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            # compute the target for endpoint parameterization
            if self.parameterization in ['endpoint', 'dirichlet', 'ctmc']:
                target = data_src[f'{feat}_1_true']
                if feat == "e":
                    target = target[upper_edge_mask]
                if feat in ['a', 'c', 'e']:
                    if self.target_blur == 0.0:
                        target = target.argmax(dim=-1)
                    else:
                        target = target + torch.randn_like(target)*self.target_blur
                        target = F.softmax(target, dim=-1)
            #  compute the target for vector-field parameterization
            elif self.parameterization == 'vector-field':
                alpha_t_prime_i = alpha_t_prime[:, feat_idx]
                x_1 = data_src[f'{feat}_1_true']
                x_0 = data_src[f'{feat}_0']

                if feat == 'e':
                    alpha_t_prime_i = alpha_t_prime_i[edge_batch_idx][upper_edge_mask].unsqueeze(-1)
                    x_1 = x_1[upper_edge_mask]
                    x_0 = x_0[upper_edge_mask]
                else:
                    alpha_t_prime_i = alpha_t_prime_i[node_batch_idx].unsqueeze(-1)

                target = alpha_t_prime_i*(x_1 - x_0)

            # for CTMC parameterization, we do not apply loss on already unmasked features
            if self.parameterization == 'ctmc' and feat in ['a', 'c', 'e']:
                if feat == 'e':
                    xt_idxs = data_src[f'{feat}_t'][upper_edge_mask].argmax(-1)
                else:
                    xt_idxs = data_src[f'{feat}_t'].argmax(-1)
                # note that we use the default ignore_index of the CrossEntropyLoss class here
                target[ xt_idxs != self.n_cat_dict[feat] ] = -100 # set the target to ignore_index when the feature is already unmasked in xt

            targets[feat] = target

        # get the time-dependent loss weights if necessary
        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)

        # compute losses
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if self.time_scaled_loss:
                weight = time_weights[:, feat_idx]
                if feat == 'e':
                    weight = weight[edge_batch_idx][upper_edge_mask]
                else:
                    weight = weight[node_batch_idx]
            else:
                weight = 1.0

            target = targets[feat]
            losses[feat] = _safe_feature_loss(
                self.loss_fn_dict[feat],
                vf_output[feat],
                target,
                weight,
                time_scaled_loss=self.time_scaled_loss,
                ctmc_categorical=(
                    self.parameterization == 'ctmc'
                    and feat in ['a', 'c', 'e']
                ),
            )

        return losses

    def sample_prior(self, g, node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        """Sample from the prior distribution of the ligand."""
        # sample atom positions from prior
        # TODO: we should set the standard deviation of atom position prior to be like the average distance to the COM in the training set
        # or perhaps the average distance to COM for molecules with the same number of atoms
        num_nodes = g.num_nodes()
        device = g.device


        # sample the prior for node features
        for feat in self.node_feats:
            prior_type = self.prior_config[feat]['type']
            prior_fn = inference_prior_register[prior_type]
            # I tried to design consistent interface for prior functions, but it's not perfect
            # hence the need for the following two if statements
            if feat == 'x':
                args = [g, node_batch_idx,]
            else:
                args = [num_nodes, self.n_cat_dict[feat],]

            if feat == 'c' and self.prior_config[feat]['type'] == 'c-given-a':
                args.append(g.ndata['a_0'])

            kwargs = self.prior_config[feat]['kwargs']
            g.ndata[f'{feat}_0'] = prior_fn(*args, **kwargs).to(device)

        # sample the prior for edge features
        g.edata['e_0'] = edge_prior(upper_edge_mask, self.prior_config['e']).to(device)

        return g

    def configure_optimizers(self):
        try:
            weight_decay = self.lr_scheduler_config['weight_decay']
        except KeyError:
            weight_decay = 0

        optimizer = optim.Adam(self.parameters(), lr=self.lr_scheduler_config['base_lr'], weight_decay=weight_decay)
        self.lr_scheduler = LRScheduler(model=self, optimizer=optimizer, **self.lr_scheduler_config)
        return optimizer

    def build_n_atoms_dist(self, n_atoms_hist_file: str):
        """Builds the distribution of the number of atoms in a ligand."""
        n_atoms, n_atom_counts = torch.load(
            n_atoms_hist_file, map_location='cpu', weights_only=False
        )
        n_atoms_prob = n_atom_counts / n_atom_counts.sum()
        self.n_atoms_dist = torch.distributions.Categorical(probs=n_atoms_prob)
        self.n_atoms_map = n_atoms

    def sample_n_atoms(self, n_molecules: int, **kwargs):
        """Draw samples from the distribution of the number of atoms in a ligand."""
        n_atoms = self.n_atoms_dist.sample((n_molecules,), **kwargs)
        return self.n_atoms_map[n_atoms]

    def sample_random_sizes(self, n_molecules: int, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None,
        xt_traj=False, ep_traj=False,
        normalization_file_path:str=None, conditional_generation:bool=True,
        property_name:str=None, # properties name is for finding normalizing vector for the property
        properties_for_sampling:int|float=None,
        training_mode:bool=True,
        properties_handle_method:str=None, # choices ['concatenate', 'sum', 'multiply', 'concatenate_sum', 'concatenate_multiply']
        multilple_values_to_one_property: List[float|int] | None = None,
        number_of_atoms: List[int]|None = None,
        **kwargs):
        """Sample molecules with sizes drawn from the training distribution."""

        if guide_w is None:
            guide_w = {'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0}
        else:
            guide_w = dict(guide_w)
        if multilple_values_to_one_property is not None and properties_for_sampling is not None:
            raise ValueError('You can not provide both multilple_values_to_one_property and properties_for_sampling, only one of them should be provided')

        # get the number of atoms that will be in each molecules
        if number_of_atoms:
            atoms_per_molecule = torch.tensor(number_of_atoms).to(device)
        else:
            atoms_per_molecule = self.sample_n_atoms(n_molecules).to(device)

        if multilple_values_to_one_property is not None:
            assert len(atoms_per_molecule) == len(multilple_values_to_one_property), \
                f"{len(atoms_per_molecule)} != {len(multilple_values_to_one_property)}"

        return self.sample(atoms_per_molecule,
            device=device,
            stochasticity=stochasticity,
            high_confidence_threshold=high_confidence_threshold,
            xt_traj=xt_traj,
            ep_traj=ep_traj,
            normalization_file_path=normalization_file_path,
            conditional_generation=conditional_generation,
            property_name=property_name, # just for sampling process with normalizing
            properties_for_sampling=properties_for_sampling,
            training_mode=training_mode,
            properties_handle_method=properties_handle_method,
            multilple_values_to_one_property=multilple_values_to_one_property,
            **kwargs)


    @torch.no_grad()
    def sample(self, n_atoms: torch.Tensor, n_timesteps: int = None, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, xt_traj=False, ep_traj=False,
        normalization_file_path:str=None, conditional_generation:bool=True,
        property_name:str=None, properties_for_sampling:int|float=None,
        training_mode:bool=True,
        properties_handle_method:str='concatenate_sum', # choices ['concatenate', 'sum', 'multiply', 'concatenate_sum', 'concatenate_multiply']
        multilple_values_to_one_property: List[float|int] | None = None,
         **kwargs):
        """Sample molecules with the given number of atoms.

        Args:
            n_atoms (torch.Tensor): Tensor of shape (batch_size,) containing the number of atoms in each molecule.
        """
        if guide_w is None:
            guide_w = {'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0}
        else:
            guide_w = dict(guide_w)
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        if xt_traj or ep_traj:
            visualize = True
        else:
            visualize = False

        batch_size = n_atoms.shape[0]

        # get the edge indicies for each unique number of atoms
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        # construct a graph for each molecule
        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
            g.append(g_i)


        # batch the graphs
        g = dgl.batch(g)

        # get upper edge mask
        upper_edge_mask = get_upper_edge_mask(g)

        # compute node_batch_idx
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

        # integrate trajectories
        integrate_kwargs = {
            'upper_edge_mask': upper_edge_mask,
            'n_timesteps': n_timesteps,
            'visualize': visualize,
            'normalization_file_path': normalization_file_path,
            'conditional_generation': conditional_generation,
            'property_name': property_name,
            'properties_for_sampling': properties_for_sampling,
            'training_mode': training_mode,
            'properties_handle_method': properties_handle_method,
            'multilple_values_to_one_property': multilple_values_to_one_property
        }
        if self.parameterization == 'ctmc':
            integrate_kwargs['stochasticity'] = stochasticity
            integrate_kwargs['high_confidence_threshold'] = high_confidence_threshold

        itg_result = self.vector_field.integrate(g, node_batch_idx, **integrate_kwargs, **kwargs)

        if visualize:
            g, traj_frames = itg_result
        else:
            g = itg_result

        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        if self.parameterization == 'ctmc':
            ctmc_mol = True
        else:
            ctmc_mol = False


        molecules = []
        for mol_idx, g_i in enumerate(dgl.unbatch(g)):

            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(SampledMolecule(*args,
                ctmc_mol=ctmc_mol,
                build_xt_traj=xt_traj,
                build_ep_traj=ep_traj,
                exclude_charges=self.exclude_charges))

        return molecules

# ========================================================================================
# Classifier-free guidance (original molguidance/models/classifier_free_guidance.py)
# ========================================================================================

class ZerosEmbedding(nn.Module):
    """
    Module that returns a tensor of zeros with the specified hidden dimension.
    Used for unconditional generation in classifier-free guidance.
    """
    def __init__(self, hidden_dim: int=256):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor|int, device='cuda') -> torch.Tensor:
        if isinstance(x, int):
            return torch.zeros(x, self.hidden_dim, device=device)
        return torch.zeros(x.size(0), self.hidden_dim, device=x.device)

class SetEmbeddingType:
    """
    Controls whether to use conditional or unconditional embeddings for each
    batch element during training.

    Similar to mattergen's SetEmbeddingType, this class creates a mask where
    True indicates using the unconditional embedding and False indicates using
    the conditional embedding.
    """
    def __init__(self, p_unconditional: float = 0.2):
        """
        Args:
            p_unconditional: Probability of using unconditional embedding during training
        """
        self.p_unconditional = p_unconditional

    def __call__(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Creates and sets the unconditional embedding mask for the graph.

        Args:
            g: DGL graph with property information

        Returns:
            DGL graph with _USE_UNCONDITIONAL_EMBEDDING attribute
        """
        # Only proceed if the graph has property information
        if not hasattr(g, 'prop') or g.prop is None:
            return g

        batch_size = g.batch_size
        device = g.device

        # Generate random mask where True = use unconditional embedding
        # Shape: [batch_size, 1]
        mask = torch.rand(batch_size, 1, device=device) <= self.p_unconditional

        # Add the mask to the graph
        g._USE_UNCONDITIONAL_EMBEDDING = mask

        return g

class ClassifierFreeGuidance(FlowMol):
    """
    Extended FlowMol model with Classifier-Free Guidance capabilities.

    During training, each batch element randomly uses either conditional
    or unconditional embeddings based on a probability mask.

    During sampling, combines conditional and unconditional predictions
    with a guidance scale to control the influence of the conditioning.
    """

    def __init__(self, *args,
                 p_uncond: float = 0.2,  # Probability of training with unconditional embedding
                 condition_margin: float = 0.0,
                 condition_margin_weight: float = 0.0,
                 **kwargs):
        if not 0.0 <= float(p_uncond) <= 1.0:
            raise ValueError(f"p_uncond must be in [0, 1], got {p_uncond}")
        if float(condition_margin) < 0.0:
            raise ValueError("condition_margin must be non-negative")
        if float(condition_margin_weight) < 0.0:
            raise ValueError("condition_margin_weight must be non-negative")
        if float(condition_margin_weight) > 0.0 and float(p_uncond) != 0.0:
            raise ValueError("condition margin training requires p_uncond=0")
        super().__init__(*args, **kwargs)
        self.p_uncond = float(p_uncond)
        self.condition_margin = float(condition_margin)
        self.condition_margin_weight = float(condition_margin_weight)
        self.save_hyperparameters({
            "p_uncond": self.p_uncond,
            "condition_margin": self.condition_margin,
            "condition_margin_weight": self.condition_margin_weight,
        })

        # Create SetEmbeddingType controller
        self.embedding_controller = SetEmbeddingType(p_unconditional=self.p_uncond)

        vector_field_config = kwargs.get("vector_field_config", {})

        # Create unconditional embedder (zero embedding)
        self.unconditional_embedder = ZerosEmbedding(hidden_dim=self.property_embedding_dim)
        self.vector_field = CFGVectorField(n_atom_types=self.n_atom_types,
                                            canonical_feat_order=self.canonical_feat_order,
                                            interpolant_scheduler=self.interpolant_scheduler,
                                            n_charges=self.n_atom_charges,
                                            n_bond_types=self.n_bond_types,
                                            exclude_charges=self.exclude_charges,
                                            property_embedding_dim=self.property_embedding_dim,
                                            property_embedder=self.property_embedder,
                                            properties_handle_method=self.properties_handle_method,
                                            conditional_generation=self.conditional_generation,
                                            dataset_name=self.dataset_name,
                                            **vector_field_config)
    def compute_batch_losses(self, g: dgl.DGLGraph, stage: str) -> Dict[str, torch.Tensor]:
        # Margin training is fully conditional; p_uncond == 0 is enforced.
        mode = 'mixed' if stage == 'train' else 'conditional'
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = None
        if g.device.type == 'cuda':
            cuda_rng_state = torch.cuda.get_rng_state(g.device)

        losses = self(
            self._local_graph_copy(g),
            condition_mode=mode,
            collect_intuitive_metrics=(stage == 'val' and self.log_intuitive_metrics),
        )
        if (
            stage == 'train'
            and self.condition_margin_weight > 0.0
            and g.batch_size > 1
        ):
            # Reuse the same time and corruption; only the property changes.
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, g.device)
            shuffled_losses = self.compute_shuffled_condition_losses(g)
            if shuffled_losses is not None:
                correct_total = self._combine_feature_losses(losses)
                shuffled_total = self._combine_feature_losses(shuffled_losses)
                losses['condition_margin'] = self.condition_margin_loss(
                    correct_total, shuffled_total, self.condition_margin
                )
        return losses



    def condition_margin_loss(
        correct_loss: torch.Tensor,
        shuffled_loss: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        """Require the correct condition to beat a mismatched condition."""
        return F.relu(correct_loss - shuffled_loss + float(margin))

    def compute_shuffled_condition_losses(
        self,
        g: dgl.DGLGraph,
    ) -> Dict[str, torch.Tensor] | None:
        if not hasattr(g, 'prop') or g.prop is None or g.batch_size <= 1:
            return None
        shuffled = g.local_var()
        shuffled.prop = torch.roll(g.prop, shifts=1, dims=0).clone()
        return self(
            shuffled,
            condition_mode='conditional',
            collect_intuitive_metrics=False,
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        condition_mode: str = 'mixed',
        collect_intuitive_metrics: bool = False,
    ):
        """Compute CTMC denoising losses under a selected conditioning policy.

        ``mixed`` applies classifier-free property dropout and is used only for
        training. ``conditional`` always uses the real property target and is
        used for validation. ``unconditional`` always uses the zero embedding.
        """
        if condition_mode not in {'mixed', 'conditional', 'unconditional'}:
            raise ValueError(f'Unknown condition_mode: {condition_mode!r}')

        batch_size = g.batch_size
        device = g.device

        # Check if the attribute loss_fn_dict exists
        if not hasattr(self, 'loss_fn_dict'):
            self.configure_loss_fns(device=g.device)

        # Get batch indices of every atom and edge
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # Create a mask which selects all of the upper triangle edges from the batched graph
        upper_edge_mask = get_upper_edge_mask(g)

        # Sample timepoints for each molecule in the batch
        t = torch.rand(batch_size).float().to(device)

        # Select conditional/unconditional embeddings explicitly. Validation
        # never uses random property dropout.
        if condition_mode == 'mixed':
            g = self.embedding_controller(g)
            use_uncond = g._USE_UNCONDITIONAL_EMBEDDING.to(device)
        elif condition_mode == 'conditional':
            use_uncond = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        else:
            use_uncond = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        # Make sure prop has correct shape
        g.prop = g.prop.unsqueeze(-1) if g.prop.dim() == 1 else g.prop
        g.prop = g.prop.to(device)

        # Move property embedder to the appropriate device
        self.property_embedder = self.property_embedder.to(device)
        self.unconditional_embedder = self.unconditional_embedder.to(device)

        # Get both conditional and unconditional embeddings
        cond_emb = self.property_embedder(g.prop)
        uncond_emb = self.unconditional_embedder(g.prop)

        # True = use unconditional, False = use the actual property target.
        prop_emb = torch.where(use_uncond, uncond_emb, cond_emb)

        # Combine with time
        t_combined = torch.cat([t.unsqueeze(-1), prop_emb], dim=-1)
        t_combined = t_combined.to(device)

        # construct interpolated molecules
        g = self.vector_field.sample_conditional_path(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)

        # forward pass for the vector field
        vf_output = self.vector_field(g, t_combined, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)

        # get the target (label) for each feature
        targets = {}
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        assert self.parameterization == "ctmc", "ClassifierFreeGuidance only supports CTMC parameterization"

        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            target = data_src[f'{feat}_1_true']
            if feat == "e":
                target = target[upper_edge_mask]
            if feat in ['a', 'c', 'e']:
                if self.target_blur == 0.0:
                    target = target.argmax(dim=-1)
                else:
                    target = target + torch.randn_like(target)*self.target_blur
                    target = F.softmax(target, dim=-1)

            # for CTMC parameterization, we do not apply loss on already unmasked features
            if feat in ['a', 'c', 'e']:
                if feat == 'e':
                    xt_idxs = data_src[f'{feat}_t'][upper_edge_mask].argmax(-1)
                else:
                    xt_idxs = data_src[f'{feat}_t'].argmax(-1)
                # note that we use the default ignore_index of the CrossEntropyLoss class here
                target[ xt_idxs != self.n_cat_dict[feat] ] = -100 # set the target to ignore_index when the feature is already unmasked in xt

            targets[feat] = target

        # get the time-dependent loss weights if necessary
        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)

        # compute losses
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if self.time_scaled_loss:
                weight = time_weights[:, feat_idx]
                if feat == 'e':
                    weight = weight[edge_batch_idx][upper_edge_mask]
                else:
                    weight = weight[node_batch_idx]
            else:
                weight = 1.0

            target = targets[feat]
            losses[feat] = _safe_feature_loss(
                self.loss_fn_dict[feat],
                vf_output[feat],
                target,
                weight,
                time_scaled_loss=self.time_scaled_loss,
                ctmc_categorical=(
                    self.parameterization == 'ctmc'
                    and feat in ['a', 'c', 'e']
                ),
            )

        if self.valence_loss_weight > 0.0:
            losses['valence'] = self._expected_valence_loss(
                g,
                vf_output['e'],
                upper_edge_mask,
            )

        if collect_intuitive_metrics:
            self._last_intuitive_validation_metrics = (
                self._build_intuitive_validation_metrics(
                    g,
                    vf_output,
                    targets,
                    upper_edge_mask,
                )
            )

        return losses

    @torch.no_grad()
    def sample(self, n_atoms: torch.Tensor, n_timesteps: int = None, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, xt_traj=False, ep_traj=False,
        normalization_file_path:str=None, conditional_generation:bool=True,
        property_name:str=None, properties_for_sampling:int|float=None,
        training_mode:bool=True,
        properties_handle_method:str='concatenate_sum',
        multilple_values_to_one_property: List[float|int] | None = None,
        guide_w: Dict[str, float] | None = None,
        dfm_type='campbell',
        guidance_format: str = "linear",  # 'linear' or 'log'
        where_to_apply_guide: str = "probabilities",  # 'probabilities' or 'rate_matrix',
        dataset_name: str = "qm9",
         **kwargs):
        """
        Sample molecules with classifier-free guidance.
        """
        if guide_w is None:
            guide_w = {'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0}
        else:
            guide_w = dict(guide_w)
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        if xt_traj or ep_traj:
            visualize = True
        else:
            visualize = False

        # Create a batched graph
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
            g.append(g_i)

        g = dgl.batch(g)
        upper_edge_mask = get_upper_edge_mask(g)
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # Sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

        # Setup integration arguments
        integrate_kwargs = {
            'upper_edge_mask': upper_edge_mask,
            'n_timesteps': n_timesteps,
            'visualize': visualize,
            'guide_w': guide_w,
            'normalization_file_path': normalization_file_path,
            'conditional_generation': conditional_generation,
            'property_name': property_name,
            'properties_for_sampling': properties_for_sampling,
            'training_mode': training_mode,
            'properties_handle_method': properties_handle_method,
            'multilple_values_to_one_property': multilple_values_to_one_property,
            'dataset_name': dataset_name
        }

        if self.parameterization == 'ctmc':
            integrate_kwargs['stochasticity'] = stochasticity
            integrate_kwargs['high_confidence_threshold'] = high_confidence_threshold
            integrate_kwargs['dfm_type'] = dfm_type
            integrate_kwargs['guidance_format'] = guidance_format
            integrate_kwargs['where_to_apply_guide'] = where_to_apply_guide

        # Integrate with guidance
        itg_result = self.vector_field.integrate_with_CFG_guidance(g, node_batch_idx, **integrate_kwargs, **kwargs)

        if visualize:
            g, traj_frames = itg_result
        else:
            g = itg_result

        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        if self.parameterization == 'ctmc':
            ctmc_mol = True
        else:
            ctmc_mol = False

        # Build molecule objects
        molecules = []
        for mol_idx, g_i in enumerate(dgl.unbatch(g)):
            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(SampledMolecule(*args,
                ctmc_mol=ctmc_mol,
                build_xt_traj=xt_traj,
                build_ep_traj=ep_traj,
                exclude_charges=self.exclude_charges))

        return molecules

    def sample_random_sizes(self, n_molecules: int, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None,
        xt_traj=False, ep_traj=False,
        normalization_file_path:str=None, conditional_generation:bool=True,
        property_name:str=None, properties_for_sampling:int|float=None,
        training_mode:bool=True,
        properties_handle_method:str=None,
        multilple_values_to_one_property: List[float|int] | None = None,
        number_of_atoms: List[int]|None = None,
        guide_w: Dict[str, float] | None = None,
        dfm_type='campbell',
        guidance_format: str = "linear",
        where_to_apply_guide: str = "probabilities",
        dataset_name: str = "qm9",
        **kwargs):
        """Sample molecules with sizes drawn from the training distribution."""

        if guide_w is None:
            guide_w = {'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0}
        else:
            guide_w = dict(guide_w)
        if multilple_values_to_one_property is not None and properties_for_sampling is not None:
            raise ValueError('You can not provide both multilple_values_to_one_property and properties_for_sampling, only one of them should be provided')

        # get the number of atoms that will be in each molecules
        if number_of_atoms:
            atoms_per_molecule = torch.tensor(number_of_atoms).to(device)
        else:
            atoms_per_molecule = self.sample_n_atoms(n_molecules).to(device)

        if multilple_values_to_one_property is not None:
            assert len(atoms_per_molecule) == len(multilple_values_to_one_property), \
                f"{len(atoms_per_molecule)} != {len(multilple_values_to_one_property)}"

        return self.sample(atoms_per_molecule,
            device=device,
            stochasticity=stochasticity,
            high_confidence_threshold=high_confidence_threshold,
            xt_traj=xt_traj,
            ep_traj=ep_traj,
            normalization_file_path=normalization_file_path,
            conditional_generation=conditional_generation,
            property_name=property_name, # just for sampling process with normalizing
            properties_for_sampling=properties_for_sampling,
            training_mode=training_mode,
            properties_handle_method=properties_handle_method,
            multilple_values_to_one_property=multilple_values_to_one_property,
            guide_w=guide_w,
            dfm_type=dfm_type,
            guidance_format=guidance_format,
            where_to_apply_guide=where_to_apply_guide,
            dataset_name=dataset_name,
            **kwargs)


class CFGVectorField(CTMCVectorField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unconditional_embedder = ZerosEmbedding(hidden_dim=self.property_embedding_dim)

    def integrate_with_CFG_guidance(self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor,
            upper_edge_mask: torch.Tensor, n_timesteps: int,
            guide_w: Dict[str, float] | None = None,
            visualize=False,
            dfm_type='campbell',
            stochasticity=8.0,
            high_confidence_threshold=0,
            cat_temp_func=None,
            forward_weight_func=None,
            tspan=None,
            normalization_file_path:str=None,
            conditional_generation:bool=True,
            property_name:str=None,
            properties_for_sampling:int|float=None,
            training_mode:bool=True,
            properties_handle_method:str=None,
            multilple_values_to_one_property: List[float|int] | None = None,
            guidance_format: str = "linear",
            where_to_apply_guide: str = "probabilities",
            dataset_name: str = "qm9",
            **kwargs):

        if guide_w is None:
            guide_w = {'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0}
        else:
            guide_w = dict(guide_w)
        self.properties_for_sampling = properties_for_sampling
        self.property_name = property_name
        self.conditional_generation = conditional_generation
        self.normalization_file_path = normalization_file_path
        self.training_mode = training_mode
        self.properties_handle_method = properties_handle_method
        self.multilple_values_to_one_property = multilple_values_to_one_property
        self.dataset_name = dataset_name

        if stochasticity is None:
            eta = self.eta
        else:
            eta = stochasticity

        if high_confidence_threshold is None:
            hc_thresh = self.hc_thresh
        else:
            hc_thresh = high_confidence_threshold

        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func
        if forward_weight_func is None:
            forward_weight_func = self.forward_weight_func

        # get edge_batch_idx
        edge_batch_idx = get_edge_batch_idxs(g)

        # get the timepoint for integration
        if tspan is None:
            t = torch.linspace(0, 1, n_timesteps, device=g.device)
        else:
            t = tspan

        # Get alpha values
        alpha_t = self.interpolant_scheduler.alpha_t(t)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # Initialize features
        for feat in self.canonical_feat_order:
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_t'] = data_src[f'{feat}_0']

        # Setup visualization if needed
        if visualize:
            traj_frames = {}
            for feat in self.canonical_feat_order:
                if feat == "e":
                    data_src = g.edata
                    split_sizes = g.batch_num_edges()
                else:
                    data_src = g.ndata
                    split_sizes = g.batch_num_nodes()

                split_sizes = split_sizes.detach().cpu().tolist()
                init_frame = data_src[f'{feat}_0'].detach().cpu()
                init_frame = torch.split(init_frame, split_sizes)
                traj_frames[feat] = [init_frame]
                traj_frames[f'{feat}_1_pred'] = []

        # Integration loop
        for s_idx in range(1, t.shape[0]):
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            dt = s_i - t_i
            last_step = (s_idx == t.shape[0] - 1)

            g = self.step_with_CFG_guidance(g, s_i, t_i,
                                        alpha_t[s_idx - 1], alpha_t[s_idx], alpha_t_prime[s_idx - 1],
                                        node_batch_idx, edge_batch_idx, upper_edge_mask,
                                        guide_w=guide_w,
                                        cat_temp_func=cat_temp_func,
                                        forward_weight_func=forward_weight_func,
                                        dfm_type=dfm_type,
                                        stochasticity=eta,
                                        high_confidence_threshold=hc_thresh,
                                        last_step=last_step,
                                        normalization_file_path=normalization_file_path,
                                        conditional_generation=conditional_generation,
                                        property_name=property_name,
                                        properties_for_sampling=properties_for_sampling,
                                        training_mode=training_mode,
                                        properties_handle_method=properties_handle_method,
                                        multilple_values_to_one_property=multilple_values_to_one_property,
                                        guidance_format=guidance_format,
                                        where_to_apply_guide=where_to_apply_guide,
                                        **kwargs)

            if visualize:
                for feat in self.canonical_feat_order:
                    if feat == "e":
                        g_data_src = g.edata
                        split_sizes = g.batch_num_edges()
                    else:
                        g_data_src = g.ndata
                        split_sizes = g.batch_num_nodes()

                    split_sizes = split_sizes.detach().cpu().tolist()
                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[feat].append(frame)

                    ep_frame = g_data_src[f'{feat}_1_pred'].detach().cpu()
                    ep_frame = torch.split(ep_frame, split_sizes)
                    traj_frames[f'{feat}_1_pred'].append(ep_frame)

        # Set final values
        for feat in self.canonical_feat_order:
            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata
            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:
            # Reshape trajectory frames
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in traj_frames.keys():
                    feat_traj = []
                    n_frames = len(traj_frames[feat])
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)

            return g, reshaped_traj_frames

        return g

    def step_with_CFG_guidance(self, g, s_i, t_i, alpha_t_i, alpha_s_i, alpha_t_prime_i,
                        node_batch_idx, edge_batch_idx, upper_edge_mask,
                        guide_w: Dict[str, float] | None = None,
                        cat_temp_func=None,
                        forward_weight_func=None,
                        dfm_type='campbell',
                        stochasticity=8.0,
                        high_confidence_threshold=0.9,
                        last_step=False,
                        normalization_file_path:str=None,
                        conditional_generation:bool=True,
                        property_name:str=None,
                        properties_for_sampling:int|float=None,
                        training_mode:bool=True,
                        properties_handle_method:str=None,
                        multilple_values_to_one_property: List[float|int] | None = None,
                        guidance_format: str = "linear",  # 'linear' or 'log'
                        where_to_apply_guide: str = "probabilities",  # 'probabilities' or 'rate_matrix'
                        **kwargs):

        if guide_w is None:
            guide_w = {'x': 2.0, 'a': 1.0, 'c': 1.0, 'e': 1.0}
        # Get predictions from unconditional and conditional models

        # Unconditional
        uncond_emb = self.unconditional_embedder(g.batch_size, device=g.device)
        t_batch = torch.full((g.batch_size,), t_i, device=g.device).unsqueeze(-1)
        t_combined_uncond = torch.cat([t_batch, uncond_emb], dim=-1)
        uncond_pred = self(g, t=t_combined_uncond,
                    node_batch_idx=node_batch_idx,
                    upper_edge_mask=upper_edge_mask,
                    apply_softmax=False, # will do softmax later
                    remove_com=True)

        # the forward function in CTMCVectorField class of ctmc_vector_field.py will take care of conditional embeddings,
        # it match case 2 of forward function adn will handle the conditional embeddings with the time t_i
        cond_pred = self(g, t=torch.full((g.batch_size,), t_i, device=g.device),
                                node_batch_idx=node_batch_idx,
                                upper_edge_mask=upper_edge_mask,
                                apply_softmax=False,
                                remove_com=True)

        dt = s_i - t_i

        # Handle positions
        x_1_cond = cond_pred['x']
        x_1_uncond = uncond_pred['x']
        x_t = g.ndata['x_t']
        guide_w_pos = guide_w['x']

        vf_cond = self.vector_field(x_t, x_1_cond, alpha_t_i[0], alpha_t_prime_i[0])
        vf_uncond = self.vector_field(x_t, x_1_uncond, alpha_t_i[0], alpha_t_prime_i[0])
        vf = (1 - guide_w_pos) * vf_uncond + guide_w_pos * vf_cond

        g.ndata['x_t'] = x_t + dt * vf
        g.ndata['x_1_pred'] = ((1 - guide_w_pos) * x_1_uncond + guide_w_pos * x_1_cond).detach().clone()

        # Handle categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'x':
                continue

            guide_weight = guide_w[feat] # Here 'feat' will be 'a', 'c', or 'e'

            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            xt = data_src[f'{feat}_t'].argmax(-1)

            if feat == 'e':
                xt = xt[upper_edge_mask]

            # Apply temperature
            temperature = cat_temp_func(t_i)

            if dfm_type == 'campbell':
                # Apply CFG directly in logit space. This is equivalent to
                # log-probability CFG after normalization, but avoids softmax
                # underflow followed by ``-inf - (-inf)`` at low temperatures.
                guided_logits = (
                    uncond_pred[feat]
                    + guide_weight * (cond_pred[feat] - uncond_pred[feat])
                )
                temperature = torch.as_tensor(
                    temperature,
                    dtype=guided_logits.dtype,
                    device=guided_logits.device,
                ).clamp_min(torch.finfo(guided_logits.dtype).eps)
                p_s_1 = F.softmax(guided_logits / temperature, dim=-1)

                xt, x_1_sampled = self.campbell_step(
                    p_1_given_t=p_s_1,
                    xt=xt,
                    stochasticity=stochasticity,
                    hc_thresh=high_confidence_threshold,
                    alpha_t=alpha_t_i[feat_idx],
                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                    dt=dt,
                    batch_size=g.batch_size,
                    batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(),
                    n_classes=self.n_cat_feats[feat]+1,
                    mask_index=self.mask_idxs[feat],
                    last_step=last_step,
                    batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx
                )

            elif dfm_type == 'campbell_rate_matrix':
                uncond_val = uncond_pred[feat]
                cond_val = cond_pred[feat]

                if where_to_apply_guide == "probabilities":
                    # not sure should we add temperature here or later yet
                    p_1_given_t_uncond = F.softmax(uncond_val, dim=-1)
                    p_1_given_t_cond = F.softmax(cond_val, dim=-1)
                elif where_to_apply_guide == "rate_matrix":
                    p_1_given_t_uncond = F.softmax(uncond_val / temperature, dim=-1)
                    p_1_given_t_cond = F.softmax(cond_val / temperature, dim=-1)

                xt, x_1_sampled = CFGVectorField.campbell_step_with_rate_matrix_cfg(
                    p_1_given_t_uncond=p_1_given_t_uncond,
                    p_1_given_t_cond=p_1_given_t_cond,
                    xt=xt,
                    stochasticity=stochasticity,
                    alpha_t=alpha_t_i[feat_idx],
                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                    dt=dt,
                    guide_weight=guide_weight,
                    mask_index=self.mask_idxs[feat],
                    n_classes=self.n_cat_feats[feat]+1,
                    uncond_val=uncond_val,
                    cond_val=cond_val,
                    last_step=last_step,
                    eps=1e-9,
                    guidance_format=guidance_format,
                    where_to_apply_guide=where_to_apply_guide,
                    temperature=temperature
                )

            # Handle edge features
            if feat == 'e':
                e_t = torch.zeros_like(g.edata['e_t'])
                e_t[upper_edge_mask] = xt
                e_t[~upper_edge_mask] = xt
                xt = e_t

                e_1_sampled = torch.zeros_like(g.edata['e_t'])
                e_1_sampled[upper_edge_mask] = x_1_sampled
                e_1_sampled[~upper_edge_mask] = x_1_sampled
                x_1_sampled = e_1_sampled

            data_src[f'{feat}_t'] = xt
            data_src[f'{feat}_1_pred'] = x_1_sampled

        return g

    @staticmethod
    def guided_probabilities_from_logits(
        uncond_logits: torch.Tensor,
        cond_logits: torch.Tensor,
        guide_weight: float,
        temperature: float | torch.Tensor,
        guidance_format: str,
    ) -> torch.Tensor:
        """Build a finite categorical CFG distribution in logit space."""
        temperature_tensor = torch.as_tensor(
            temperature,
            dtype=uncond_logits.dtype,
            device=uncond_logits.device,
        ).clamp_min(torch.finfo(uncond_logits.dtype).eps)
        if guidance_format == "log":
            log_p_uncond = F.log_softmax(uncond_logits, dim=-1)
            log_p_cond = F.log_softmax(cond_logits, dim=-1)
            guided_logits = (
                log_p_uncond
                + guide_weight * (log_p_cond - log_p_uncond)
            )
        elif guidance_format == "linear":
            guided_logits = (
                uncond_logits
                + guide_weight * (cond_logits - uncond_logits)
            )
        else:
            raise ValueError(
                f"Invalid guidance_format: {guidance_format}. "
                'Choose "linear" or "log".'
            )
        return F.softmax(guided_logits / temperature_tensor, dim=-1)

    @staticmethod
    def campbell_step_with_rate_matrix_cfg(p_1_given_t_uncond: torch.Tensor,
                                        p_1_given_t_cond: torch.Tensor,
                                        xt: torch.Tensor,
                                        stochasticity: float,
                                        alpha_t: float,
                                        alpha_t_prime: float,
                                        dt: float,
                                        guide_weight: float,
                                        mask_index: int,
                                        n_classes: int,
                                        uncond_val: torch.Tensor,
                                        cond_val: torch.Tensor,
                                        last_step: bool = False,
                                        eps: float = 1e-9,
                                        guidance_format: str = "linear",
                                        where_to_apply_guide: str = "probabilities",
                                        temperature: float = 1.0
                                        ):
        """
        Modified campbell_step that applies CFG to rate matrices (paper's approach)
        Args:
            p_1_given_t_uncond: Unconditional model predictions probabilities
            p_1_given_t_cond: Conditional model predictions probabilities
            xt: Current state indices [N]
            guide_weight: CFG guidance weight
            uncond_val: Unconditional model's prediction value before applying softmax converted to probabilities
            cond_val: Conditional model's prediction value before applying softmax converted to probabilities
            last_step: Whether this is the last step of integration
            eps: Small value to avoid log(0)
            guidance_format: 'linear' or 'log' for combining rate matrices
            where_to_apply_guide: 'probabilities' or 'rate_matrix' to apply guidance
        """
        device = xt.device
        # print(f"xt shape: {xt.shape}\n")

        if where_to_apply_guide == "rate_matrix":
            # Step 1: Compute separate rate matrices for unconditional and conditional
            R_t_uncond = CFGVectorField._compute_rate_matrix(p_1_given_t_uncond, xt, alpha_t, alpha_t_prime,
                                                stochasticity, mask_index, n_classes)
            R_t_cond = CFGVectorField._compute_rate_matrix(p_1_given_t_cond, xt, alpha_t, alpha_t_prime,
                                                stochasticity, mask_index, n_classes)

            if guidance_format == "log":
                # Step 2: Combine rate matrices with guidance
                R_t_guided = torch.exp(
                    (1 - guide_weight) * torch.log(R_t_uncond + eps) +
                    guide_weight * torch.log(R_t_cond + eps)
                )
            elif guidance_format == "linear":
                R_t_guided = (1 - guide_weight) * R_t_uncond + guide_weight * R_t_cond
            else:
                raise ValueError(f"Invalid guidance_format: {guidance_format}. Choose 'linear' or 'log'.")

            # Extrapolative CFG weights can make a linear rate combination
            # negative. Off-diagonal CTMC rates must remain non-negative.
            R_t_guided = R_t_guided.clamp_min(0.0)

        elif where_to_apply_guide == "probabilities":
            p_s_1 = CFGVectorField.guided_probabilities_from_logits(
                uncond_val,
                cond_val,
                guide_weight,
                temperature,
                guidance_format,
            )
            R_t_guided = CFGVectorField._compute_rate_matrix(p_s_1, xt, alpha_t, alpha_t_prime,
                                                stochasticity, mask_index, n_classes)

        # Step 3: Re-normalize rate matrix (diagonal = -row_sum)
        R_t_guided.scatter_(-1, xt.unsqueeze(-1), 0.0)  # Clear diagonal
        row_sums = R_t_guided.sum(dim=-1, keepdim=True)
        R_t_guided.scatter_(-1, xt.unsqueeze(-1), -row_sums)  # Set diagonal

        # Step 4: Convert to transition probabilities
        step_probs = (R_t_guided * dt).clamp(min=0.0, max=1.0)

        # Ensure probability conservation
        step_probs.scatter_(-1, xt.unsqueeze(-1), 0.0)
        stay_probs = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, xt.unsqueeze(-1), stay_probs)
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        # Step 5: Handle last step
        if last_step:
            # For last step, create guided logits with S+1 dimensions
            guided_logits_full = torch.zeros(uncond_val.shape[0], n_classes, device=device)
            guided_logits_full[:, :uncond_val.shape[-1]] = (1 - guide_weight) * uncond_val + guide_weight * cond_val
            guided_logits_full[:, mask_index] = -1e9 # Never choose mask in final step

            is_masked = (xt == mask_index)
            xt_new = xt.clone()
            xt_new[is_masked] = guided_logits_full[is_masked].argmax(-1)
        else:
            xt_new = Categorical(step_probs).sample()

        LARGE_NEG = -1e9

        log_p_uncond_full = torch.full((p_1_given_t_uncond.shape[0], n_classes), LARGE_NEG, device=device)
        log_p_cond_full = torch.full((p_1_given_t_cond.shape[0], n_classes), LARGE_NEG, device=device)

        log_p_uncond_full[:, :p_1_given_t_uncond.shape[-1]] = torch.log(p_1_given_t_uncond + eps)
        log_p_cond_full[:, :p_1_given_t_cond.shape[-1]] = torch.log(p_1_given_t_cond + eps)

        log_p_guided = (1 - guide_weight) * log_p_uncond_full + guide_weight * log_p_cond_full # just for x1 for visualization

        # Debug: Check for NaN before softmax
        if torch.isnan(log_p_guided).any():
            print("Warning: NaN detected in log_p_guided before softmax!")
            print(f"guide_weight: {guide_weight}")
            print(f"log_p_uncond_full stats: min={log_p_uncond_full.min()}, max={log_p_uncond_full.max()}, has_nan={torch.isnan(log_p_uncond_full).any()}")
            print(f"log_p_cond_full stats: min={log_p_cond_full.min()}, max={log_p_cond_full.max()}, has_nan={torch.isnan(log_p_cond_full).any()}")

        x1 = Categorical(F.softmax(log_p_guided, dim=-1)).sample()

        xt_new = F.one_hot(xt_new, num_classes=n_classes).float()
        x1 = F.one_hot(x1, num_classes=n_classes).float()

        return xt_new, x1

    @staticmethod # adapted from https://github.com/hnisonoff/discrete_guidance/blob/main/src/fm_utils.py
    def _compute_rate_matrix(p_1_given_t: torch.Tensor, xt: torch.Tensor,
                            alpha_t: float, alpha_t_prime: float, stochasticity: float,
                            mask_index: int, n_classes: int) -> torch.Tensor:
        """
        Compute rate matrix R_t following the paper's approach
        """
        device = p_1_given_t.device
        N = xt.shape[0] # N.shape = B * D, where B is batch size and D is number of nodes/edges
        S_actual = p_1_given_t.shape[-1]  # Number of actual classes (S)

        # Create full rate matrix for all S+1 classes
        R_t = torch.zeros(N, n_classes, device=device)  # Shape: [N, S+1]

        # Masks for current state
        is_masked = (xt == mask_index).float().unsqueeze(-1)  # [N, 1]
        is_unmasked = 1 - is_masked

        # Unmasking rates: from mask to any non-mask state
        # Rate = p(x1=j|xt) * (alpha_t_prime + stochasticity*alpha_t) / (1 - alpha_t)
        unmasking_factor = (alpha_t_prime + stochasticity * alpha_t) / (1 - alpha_t + 1e-8)

        # Only fill the actual class positions (0 to S-1), leave mask position (S) as 0
        R_t[:, :S_actual] = is_masked * p_1_given_t * unmasking_factor

        # Remasking rates: from actual classes to mask token
        # For nodes that are currently unmasked, add rate to transition to mask
        R_t[:, mask_index] = is_unmasked.squeeze(-1) * stochasticity

        return R_t

# ========================================================================================
# Configuration factory
# ========================================================================================


def _scalar_from_config(config: dict, *path: str, default=None):
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def model_from_config(config: dict, seed_ckpt: str | Path | None = None) -> ClassifierFreeGuidance:
    """Instantiate the S1-conditioned CFG model from resolved runtime config."""
    guidance = config.get('guidance', {})
    if guidance.get('guidance_type', 'classifier_free_guidance') != 'classifier_free_guidance':
        raise NotImplementedError(
            'This implementation supports guidance_type=classifier_free_guidance only.'
        )

    dataset = config['dataset']
    model_setting = config.get('model_setting', {})
    processed_data_dir = Path(dataset['processed_data_dir']).expanduser().resolve()
    atom_map = dataset['atom_map']
    if atom_map is None or (
        isinstance(atom_map, str)
        and atom_map.strip().lower() in {'', 'auto', 'from_data', 'from-data', 'pt', 'processed_pt'}
    ):
        raise ValueError(
            'dataset.atom_map has not been materialized. Resolve it from the processed '
            'PT metadata before calling model_from_config.'
        )
    if isinstance(atom_map, dict):
        if atom_map and all(isinstance(value, int) for value in atom_map.values()):
            atom_type_map = [
                symbol for symbol, _ in sorted(atom_map.items(), key=lambda item: item[1])
            ]
        elif atom_map and all(str(key).isdigit() for key in atom_map):
            atom_type_map = [atom_map[key] for key in sorted(atom_map, key=lambda key: int(key))]
        else:
            raise ValueError('dataset.atom_map dict must map element->index or index->element')
    else:
        atom_type_map = list(atom_map)

    conditioning = dataset.get('conditioning', {})
    property_names = conditioning.get('property_names')
    conditioning_property = conditioning.get('property')
    if (
        conditioning_property
        and isinstance(property_names, (list, tuple))
        and conditioning_property in property_names
    ):
        PROPERTY_MAP[conditioning_property] = list(property_names).index(
            conditioning_property
        )

    validation_cfg = config.get('training', {}).get('validation', {})

    common_kwargs = {
        'atom_type_map': atom_type_map,
        'n_atoms_hist_file': processed_data_dir / 'train_data_n_atoms_histogram.pt',
        'marginal_dists_file': processed_data_dir / 'train_data_marginal_dists.pt',
        'sample_interval': _scalar_from_config(config, 'training', 'evaluation', 'sample_interval', default=1.0),
        'n_mols_to_sample': _scalar_from_config(config, 'training', 'evaluation', 'mols_to_sample', default=64),
        'vector_field_config': config.get('vector_field', {}),
        'interpolant_scheduler_config': config.get('interpolant_scheduler', {}),
        'lr_scheduler_config': config.get('lr_scheduler', {}),
        'property_embedding_dim': model_setting.get('property_embedding_dim', 256),
        'gaussian_expansion': model_setting.get('gaussian_expansion', {}).get('enabled', False),
        'gaussian_start': model_setting.get('gaussian_expansion', {}).get('start'),
        'gaussian_stop': model_setting.get('gaussian_expansion', {}).get('stop'),
        'n_gaussians': model_setting.get('gaussian_expansion', {}).get('n_gaussians', 5),
        'conditional_generation': conditioning.get('enabled', True),
        'properties_handle_method': model_setting.get('properties_handle_method', 'concatenate_sum'),
        'dataset_name': dataset.get('dataset_name', 'csvmol'),
        'conditioning_property': conditioning_property,
        'property_transform': conditioning.get('target_transform'),
        'property_epsilon': conditioning.get('target_epsilon', 1.0e-8),
        'property_normalization_file': str(
            processed_data_dir / 'train_data_property_normalization.pt'
        ),
        'validation_seed': validation_cfg.get('seed', 12345),
        'compare_shuffled_condition': validation_cfg.get(
            'compare_shuffled_condition', False
        ),
        'charge_offset': dataset.get('charge_offset', 2),
        'log_intuitive_metrics': validation_cfg.get('log_intuitive_metrics', True),
        'p_uncond': guidance.get('p_uncond', 0.2),
        'condition_margin': guidance.get('condition_margin', 0.0),
        'condition_margin_weight': guidance.get('condition_margin_weight', 0.0),
        **config.get('mol_fm', {}),
    }

    if seed_ckpt is None:
        return ClassifierFreeGuidance(**common_kwargs)
    return ClassifierFreeGuidance.load_from_checkpoint(
        str(seed_ckpt),
        map_location='cpu',
        weights_only=False,
        **common_kwargs,
    )

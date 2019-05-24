import numpy as np
import networkx as nx
from LSI_filters import random_arma

from scipy.linalg import solve_discrete_lyapunov

# TODO: It would probably be wise to do this with scipy sparse matrices!


def gcg_to_var(G, filter_attr="b(z)", assert_stable=True, p=None):
    """
    Convert a GCG to a VAR system.

    G should be an nx.DiGraph having a "max_lag" property and "b(z)"
    properties on it's edges.  i.e. G.graph["max_lag"] and G[i][j]["b(z)"]
    where G[i][j]["b(z)"] = b is an LSI filter j --b(z)--> i and
    b represents b(z) = b[0]z^{-1} + b[1]z^{-2} + ... + b[p - 1]z^{-p}
    where p = max_lag and each b should have this same length in common.

    This is implemented by walking over the graph's nodes and building
    system B(tau) matrices for the VAR class.
    """
    nodes = G.nodes
    n = len(nodes)
    if p is None:
        i, j = next(iter(G.edges))
        p = len(G[i][j][filter_attr])

    # Number the nodes
    node_map = {node_i: i for i, node_i in enumerate(nodes)}

    # The system matrices
    B = np.dstack([np.zeros((n, n)) for _ in range(p)])

    # Collect all the data
    for i, node_i in enumerate(nodes):  # Driven node
        for node_j in G.predecessors(node_i):  # Driving nodes
            j = node_map[node_j]
            try:  # Get the filter from G
                b_ij = G[node_j][node_i][filter_attr]
            except KeyError:
                raise KeyError("Each edge must have a '{}' attr!  "
                               "It is missing for edge ({}, {})"
                               "".format(filter_attr, node_i, node_j))

            try:  # Put it in B
                B[i, j, :] = b_ij
            except ValueError as e:
                raise ValueError("Caught value error: {} -- is each filter of "
                                 "the same size?  max_lag = {}, len(b_ij) = {}"
                                 "".format(e, p, len(b_ij)))

    B = [B[:, :, tau] for tau in range(p)]
    var = VAR(B)
    if assert_stable:
        assert var.is_stable()
    return var


def random_gnp_dag(n_nodes, p_lags, edge_prob=0.3, pole_rad=0.95):
    """
    Produces a random DAG on n nodes and populates the edge b(z)
    attributes with random AR(p) filters.

    we have a graph on n_nodes with s_edges and p_lags for the AR
    systems.
    """
    G = nx.gnp_random_graph(n_nodes, p=edge_prob, directed=True)

    def get_bz():
        return -random_arma(p=p_lags, q=0, k=1, p_radius=pole_rad)[1][1:]

    G = nx.DiGraph([(u, v, {"b(z)": get_bz()})
                    for (u, v) in G.edges() if u < v])
    [G.add_node(v) for v in set(range(n_nodes)) - set(G.nodes)]
    G.graph["true_lags"] = p_lags
    assert nx.is_directed_acyclic_graph(G)

    # Add self-loops
    for i in G.nodes:
        G.add_edge(i, i, **{"b(z)": get_bz()})
    return G


def get_node_property_vector(G, prop):
    """
    Returns a numpy array containing G.nodes[i][prop] in order by
    G.nodes.
    """
    try:
        data = np.array([G.nodes[i][prop] for i in G.nodes])
    except KeyError as e:
        raise KeyError("Caught KeyError({}) -- does G have "
                       "{} attached to it's nodes?"
                       "".format(e, prop))
    return data


def get_edge_property_dict(G, prop):
    """
    Returns a dict {(i, j): G[i][j][prop]}
    """
    try:
        return {(i, j): G[i][j][prop] for (i, j) in G.edges}
    except KeyError as e:
        raise KeyError("Caught KeyError({}) -- does G have "
                       "{} attached to it's edges?"
                       "".format(e, prop))


def get_errors(G):
    return get_node_property_vector(G, "sv^2_hat")


def get_estimation_errors(G):
    b = get_edge_property_dict(G, "b(z)")
    b_hat = get_edge_property_dict(G, "b_hat(z)")

    def match_len_sub(x, y):
        len_max = max(len(x), len(y))
        s = np.zeros(len_max)
        s[:len(x)] = x
        s[:len(y)] = s[:len(y)] - y
        return s

    return [match_len_sub(b[(i, j)], b_hat[(i, j)]) for (i, j) in G.edges]


def drive_gcg(G, T, sigma2_v, filter_attr="b(z)"):
    """
    Drives the gcg G with T samples of Gaussian noise.  sigma_v should
    be a sequence of variances for the noise.

    We will augment G with G.edges[i]["x"] = X[:, i] and return i.e. we
    attach the generated data to G.
    """
    n = len(G.nodes)
    V = np.random.normal(size=(T, n)) * np.sqrt(sigma2_v)
    var = gcg_to_var(G, filter_attr=filter_attr)
    X = var.drive(V)

    for i, node_i in enumerate(G.nodes):
        G.nodes[node_i]["x"] = X[:, i]
        G.nodes[node_i]["sv2"] = sigma2_v[i]
    return G


class VAR:
    '''VAR(p) system'''
    def __init__(self, B, x_0=None):
        '''Initializes the model with a list of coefficient matrices
        B = [B(1), B(2), ..., B(p)] where each B(\tau) \in \R^{n \times n}

        x_0 can serve to initialize the system output (if len(x_0) == n)
        or the entire system state (if len(x_0) == n * p)
        '''
        self.p = len(B)
        self.n = (B[0].shape[0])
        if not all(
                len(B_tau.shape) == 2 and  # Check B(\tau) is a matrix
                B_tau.shape[0] == self.n and  # Check compatible sizes
                B_tau.shape[1] == self.n  # Check square
                for B_tau in B):
            raise ValueError('Coefficients must be square matrices of '
                             'equivalent sizes')
        self.B = B  # Keep the list of matrices
        self._B = np.hstack(B)  # Standard form layout \hat{x(t)} = B^\T z(t)
        self.t = 0

        self.reset(x_0=x_0)  # Reset system state
        return

    def get_induced_graph(self):
        '''Returns the adjacency matrix of the Granger-causality graph
        induced by this VAR model.  This graph is defined via:

        G_{ij} = 1 if \exists tau s.t. B(tau)_{ji} \ne 0, and G_{ij} = 0
        otherwise.

        The interpretation of B(tau)_{ji} is as the coefficient transferring
        energy from process i to process j with a lag of tau seconds.  And,
        G_{ij} = 1 if there is some transfer from process i to j.  Hence,
        we are looking at transposes of coefficient matrices to get to
        the graph adjacency matrix.
        '''
        # We are careful to return np arrays with float type as
        # True/False do not always behave in the same way as 1./0.
        return np.array(sum(B_tau != 0 for B_tau in self.B) != 0,
                        dtype=np.float64)

    def get_companion(self):
        '''Return the companion matrix for the system

        C =
        [B0, B1, B2, ... Bp-1]
        [ I,  0,  0, ... 0   ]
        [ 0,  I,  0, ... 0   ]
        [ 0,  0,  I, ... 0   ]
        [ 0,  0, ..., I, 0   ]
        '''
        n, p = self.n, self.p
        C = np.hstack((np.eye(n * (p - 1)),  # The block diagonal I
                       np.zeros((n * (p - 1), n))))  # right col
        C = np.vstack((np.hstack((B_tau for B_tau in self.B)),  # top row
                       C))
        return C

    def is_stable(self, margin=1e-6):
        '''Check whether the system is stable.  See also self.get_rho().
        We return True if |\lambda_max(C)| <= 1 - margin.  Note that the
        default margin is very small.
        '''
        rho = self.get_rho()
        return rho <= 1 - margin

    def get_rho(self):
        '''Computes and returns the stability coefficient rho.  In order to do
        this we directly calculate the eigenvalues of the block companion
        matrix induced by B, which is of size (np x np).  This may be
        prohibitive for very large systems.

        Stability is determined by the spectral radius of the matrix:

        C =
        [B0, B1, B2, ... Bp-1]
        [ I,  0,  0, ... 0   ]
        [ 0,  I,  0, ... 0   ]
        [ 0,  0,  I, ... 0   ]
        [ 0,  0, ..., I, 0   ]

        .  Note that the
        default margin is very small.
        '''
        C = self.get_companion()
        ev = np.linalg.eigvals(C)  # Compute the eigenvalues
        return max(abs(ev))

    def acf(self, Sigma_u=None, h_lags=0):
        '''Solve the Yule-Walker equations for the autocovariance functions
        R(0), ..., R(p), assuming the system is driven by white noise with
        variance Sigma_u

        params:
          Sigma_u (default np.eye(self.n)): Variance of driving noise
          h_lags (default 1): Number of lags to compute
        returns:
          [R(0), R(1), ..., R(h_lags)] covaraince functions
        '''
        n, p = self.n, self.p
        if Sigma_u is None:
            Sigma_u = np.eye(n)
        if Sigma_u.shape != (n, n):
            raise ValueError('Sigma_u is not compatible with system dimension')
        C = self.get_companion()
        Sigma_U = np.zeros_like(C)
        Sigma_U[:n, :n] = Sigma_u
        R0 = solve_discrete_lyapunov(C, Sigma_U)

        # Compute acf
        R = []
        for tau in range(p):
            R.append(R0[:n, tau * n: (tau + 1) * n])  # R(0), ..., R(p - 1)

        for h in range(p, h_lags + 1):
            Rh = np.zeros((n, n))
            for tau in range(p):
                Rh += np.dot(self.B[tau], R[h - tau - 1])
            R.append(Rh)
        return R[:h_lags + 1]

    def reset(self, x_0=None, reset_t=True):
        '''Reset the system to some initial state.  If x_0 is specified,
        it may be of dimension n or n * p.  If it is dimension n, we simply
        dictate the value of the current output, otherwise we reset the
        whole system state.  If reset_t is True then we set the current
        time to reset_t'''
        n, p = self.n, self.p
        if x_0 is not None:
            if len(x_0) == n:  # Initialize just the output
                self._z = np.zeros(n * p)
                self._z[:n] = x_0
            elif len(x_0) == n * p:  # Initialize whole state
                self._z = x_0
            else:
                raise ValueError('Dimension %d of x_0 is not compatible with '
                                 'system dimensions n = %d, p = %d' %
                                 (len(x_0), n, p))

        else:
            self._z = np.zeros(n * p)
        self.x = self._z[:n]  # System output
        if reset_t:
            self.t = 0
        return

    def run(self, T, mu_u=None, Sigma_u=None):
        '''Drives the system with T samples of iid Gaussian noise having
        mean mu_u and variance Sigma_u.

        params:
          T (int): The number of samples to drive the system for
          mu_u (np.array): A mean vector for the system noise.
            default = np.zeros(self.n)
          Sigma_u (np.array): A variance matrix for the system noise.
            default = np.eye(self.n)

        throws:
          np.linalg.LinAlgError: If Sigma_u is not positive definite
        '''
        if mu_u is None:
            mu_u = np.zeros(self.n)

        if Sigma_u is None:
            Sigma_u = np.eye(self.n)

        assert mu_u.size == self.n, 'mu_u dimension is not compatible!'
        assert Sigma_u.shape == (self.n, self.n), 'Sigma_u dimension is not '\
            'compatible!'

        L_u = np.linalg.cholesky(Sigma_u)
        U = np.random.randn(T * self.n).reshape(T, self.n)
        U = mu_u + np.dot(U, L_u.T)
        return self.drive(U)

    def drive(self, u):
        '''
        Drives the system with input u.  u should be a T x n array
        containing a sequence of T inputs, or a single length n input.
        '''
        n, p = self.n, self.p
        if len(u.shape) == 1:  # A single input
            try:
                u = u.reshape((1, n))  # Turn it into a vector
            except ValueError:
                raise ValueError('The length %d of u is not compatible with '
                                 'system dimensions n = %d, p = %d'
                                 % (len(u), n, p))

        if u.shape[1] != n:  # Check dimensions are compatible
            raise ValueError('The dimension %d of the input vectors is '
                             'not compatible with system dimensions n = %d, '
                             ' p = %d' % (u.shape[1], n, p))

        T = u.shape[0]  # The number of time steps
        self.t += T

        # Output matrix to be returned
        Y = np.empty((T, n))
        for t in range(T):
            y = np.dot(self._B, self._z) + u[t, :]  # Next output
            Y[t, :] = y
            self._z = np.roll(self._z, n)  # Update system state
            self._z[:n] = y
        self.x = self._z[:n]  # System output
        return Y

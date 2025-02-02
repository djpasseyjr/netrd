"""
SIS.py
------

Implementation of Susceptible-Infected-Susceptible models dynamics on a
network.

author: Stefan McCabe

Submitted as part of the 2019 NetSI Collabathon.

"""

from netrd.dynamics import BaseDynamics
import numpy as np
import networkx as nx


class SISModel(BaseDynamics):
    """Susceptible-Infected-Susceptible dynamical process."""

    def simulate(self, G, L, num_seeds=1, beta=None, mu=None):
        r"""Simulate SIS model dynamics on a network.

        The results dictionary also stores the ground truth network as
        `'ground_truth'`.

        Parameters
        ----------
        G (nx.Graph)
            the input (ground-truth) graph with :math:`N` nodes.

        L (int)
            the length of the desired time series.

        num_seeds (int)
            the number of initially infected nodes.

        beta (float)
            the infection rate for the SIS process.

        mu (float)
            the recovery rate for the SIS process.

        Returns
        -------
        TS (np.ndarray)
            an :math:`N \times L` array of synthetic time series data.

        """
        H = G.copy()
        N = H.number_of_nodes()
        TS = np.zeros((N, L))
        index_to_node = dict(zip(range(G.order()), list(G.nodes())))

        # sensible defaults for beta and mu
        if not beta:
            avg_k = np.mean(list(dict(H.degree()).values()))
            beta = 1 / avg_k
        if not mu:
            mu = 1 / H.number_of_nodes()

        seeds = np.random.permutation(
            np.concatenate([np.repeat(1, num_seeds), np.repeat(0, N - num_seeds)])
        )
        TS[:, 0] = seeds
        infected_attr = {index_to_node[i]: s for i, s in enumerate(seeds)}
        nx.set_node_attributes(H, infected_attr, 'infected')
        nx.set_node_attributes(H, 0, 'next_infected')

        # SIS dynamics
        for t in range(1, L):
            # If the epidemic ended in the previous timestep, then break.
            if TS[:, t - 1].sum() < 1:
                break

            # Visit each node in a random order
            nodes = np.random.permutation(H.nodes)
            for i in nodes:
                # Visit all neighbors of each infected node and
                # randomly infect them with probability of beta
                if H.nodes[i]['infected']:
                    neigh = H.neighbors(i)
                    for j in neigh:
                        if np.random.random() < beta:
                            H.nodes[j]['next_infected'] = 1

                    # Cure the current infected node with probability mu
                    if np.random.random() < mu:
                        H.nodes[i]['infected'] = 0
                    else:
                        # If the node is not cured assign it to be
                        # infected in the next timestep
                        H.nodes[i]['next_infected'] = 1
            next_infections = nx.get_node_attributes(H, 'next_infected')
            # Assign 'next_infected' nodes to become the currently infected
            nx.set_node_attributes(H, next_infections, 'infected')
            # Reset 'next_infected' to zero for all nodes
            nx.set_node_attributes(H, 0, 'next_infected')

            # store the new infected nodes for time t
            infections = nx.get_node_attributes(H, 'infected')
            TS[:, t] = np.array(list(infections.values()))

        # if the epidemic died off, pad the time series to the right shape
        if TS.shape[1] < L:
            TS = np.hstack([TS, np.zeros((N, L - TS.shape[1]))])

        self.results['ground_truth'] = H
        self.results['TS'] = TS
        self.results['index_to_node'] = index_to_node

        return TS

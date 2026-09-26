"""Graph-based spin lattices with row-major logical site labels."""

import networkx as nx
from abc import ABC, abstractmethod


class Lattice(ABC):
    """Abstract lattice backed by a NetworkX graph."""

    def __init__(self) -> None:
        """Build the graph using the subclass's lattice specification."""
        super().__init__()
        self.graph = self.generate_graph()

    @abstractmethod
    def generate_graph(self) -> nx.Graph:
        """Return the graph whose nodes are sites and edges are interactions."""
        pass

    @property
    def num_nodes(self) -> int:
        """Return the number of spin sites in the graph."""
        return self.graph.number_of_nodes()

    @property
    def interaction_map(self) -> list:
        """Return graph edges as pairs of logical site labels."""
        return list(self.graph.edges())

    @property
    def onsite_field_map(self) -> list:
        """Return logical site labels in graph insertion order."""
        return list(self.graph.nodes())


class SquareLattice(Lattice):
    """Rectangular nearest-neighbour lattice, including one-row chains."""

    def __init__(self, Lx: int, Ly: int, periodic: bool = False) -> None:
        """Set dimensions and construct the lattice graph.

        Args:
            Lx: Number of sites along the x direction.
            Ly: Number of sites along the y direction.
            periodic: Whether to connect opposing boundaries in each
                direction with more than one site.
        """
        self.Lx = Lx
        self.Ly = Ly
        self.periodic = periodic
        super().__init__()

    def generate_graph(self) -> nx.Graph:
        """Build the graph with site labels ``x + y * Lx`` and positions.

        The undirected graph stores each edge once, including when a periodic
        boundary would duplicate an existing nearest-neighbour edge.
        """
        G_orig = nx.grid_2d_graph(self.Lx, self.Ly)
        G = nx.Graph()
        # Add nodes
        for x, y in G_orig.nodes():
            G.add_node(x + y * self.Lx, pos=(x, y))
        # Add edges
        for u, v in G_orig.edges():
            G.add_edge(u[0] + u[1] * self.Lx, v[0] + v[1] * self.Lx)
        # Add periodic boundary conditions
        if self.periodic:
            for x in range(self.Lx):
                if self.Ly > 1:
                    G.add_edge(x, x + (self.Ly - 1) * self.Lx)

            for y in range(self.Ly):
                if self.Lx > 1:
                    G.add_edge(y * self.Lx, (self.Lx - 1) + y * self.Lx)
        return G

"""Qiskit spin Hamiltonians used by independent small-benchmark scoring.

Couplings are signed coefficients of Pauli operators, not spin operators
divided by two. Logical site labels are inserted left-to-right into Qiskit
Pauli strings, whose rightmost character acts on qubit zero.
"""

from lattice import Lattice
from qiskit.quantum_info import SparsePauliOp
from abc import ABC, abstractmethod


class SpinHamiltonian(ABC):
    """Abstract lattice Hamiltonian stored as a Qiskit sparse Pauli operator."""

    def __init__(self, lattice: Lattice):
        """Store the lattice and build the operator using subclass couplings.

        Args:
            lattice: Lattice defining interaction edges and onsite terms.
        """
        self.lattice = lattice
        self.qubit_op = self.build_qubit_op()

    @abstractmethod
    def build_qubit_op(self) -> SparsePauliOp:
        """Build the Hamiltonian as a Qiskit sparse Pauli operator."""
        pass


class IsingModel(SpinHamiltonian):
    """Mixed-field Ising Hamiltonian with signed ZZ, Z, and X coefficients.

    The convention is ``H = j*sum_edges(ZZ) + hz*sum_sites(Z)
    + hx*sum_sites(X)``, with each lattice edge counted once.
    """

    def __init__(self, lattice: Lattice, j: float, hz: float, hx: float):
        """Set the signed couplings and build the Ising operator.

        Args:
            lattice: Lattice defining interaction edges and onsite terms.
            j: Signed coefficient of each ZZ edge interaction.
            hz: Signed longitudinal Z-field coefficient.
            hx: Signed transverse X-field coefficient.
        """
        self.j = j
        self.hz = hz
        self.hx = hx
        super().__init__(lattice)

    def build_qubit_op(self) -> SparsePauliOp:
        """Build and simplify the ZZ interactions and onsite Z/X fields."""
        op_list = []
        for m, n in self.lattice.interaction_map:
            str = "I" * self.lattice.num_nodes
            str = str[:m] + "Z" + str[m + 1 :]
            str = str[:n] + "Z" + str[n + 1 :]
            op_list.append((str, self.j))
        for m in self.lattice.onsite_field_map:
            str_z = "I" * self.lattice.num_nodes
            str_z = str_z[:m] + "Z" + str_z[m + 1 :]
            op_list.append((str_z, self.hz))
            str_x = "I" * self.lattice.num_nodes
            str_x = str_x[:m] + "X" + str_x[m + 1 :]
            op_list.append((str_x, self.hx))

        ham = SparsePauliOp.from_list(op_list)
        return ham.simplify()


class XXZModel(SpinHamiltonian):
    """XXZ Hamiltonian with signed Pauli-interaction coefficients.

    The convention is ``H = jxy*sum_edges(XX + YY) + jz*sum_edges(ZZ)``.
    """

    def __init__(self, lattice: Lattice, jxy: float, jz: float):
        """Set the signed couplings and build the XXZ operator.

        Args:
            lattice: Lattice defining interaction edges.
            jxy: Shared signed coefficient of XX and YY interactions.
            jz: Signed coefficient of ZZ interactions.
        """
        self.jxy = jxy
        self.jz = jz
        super().__init__(lattice)

    def __str__(self) -> str:
        """Return the model identifier."""
        return "xxz"

    def build_qubit_op(self) -> SparsePauliOp:
        """Build and simplify the XX, YY, and ZZ edge interactions."""
        op_list = []
        for m, n in self.lattice.interaction_map:
            str = "I" * self.lattice.num_nodes
            str = str[:m] + "X" + str[m + 1 :]
            str = str[:n] + "X" + str[n + 1 :]
            op_list.append((str, self.jxy))
            str = "I" * self.lattice.num_nodes
            str = str[:m] + "Y" + str[m + 1 :]
            str = str[:n] + "Y" + str[n + 1 :]
            op_list.append((str, self.jxy))
            str = "I" * self.lattice.num_nodes
            str = str[:m] + "Z" + str[m + 1 :]
            str = str[:n] + "Z" + str[n + 1 :]
            op_list.append((str, self.jz))
        ham = SparsePauliOp.from_list(op_list)
        return ham.simplify()

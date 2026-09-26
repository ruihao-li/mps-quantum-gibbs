"""Qiskit purification circuits used for independent endpoint reconstruction."""

from lattice import Lattice
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import ParameterVector
from abc import ABC, abstractmethod
from numbers import Integral


class Ansatz(ABC):
    """Abstract variational circuit ansatz on a logical spin lattice."""

    def __init__(self, lattice: Lattice):
        """Store the lattice and initialize mutable circuit settings.

        Args:
            lattice: Lattice defining the logical system sites.
        """
        self.lattice = lattice
        self.num_ancillas = None
        self.num_layers = 1

    @abstractmethod
    def build_circuit(self) -> QuantumCircuit:
        """Build an unbound Qiskit circuit from the current settings."""
        pass

    @property
    def circuit(self) -> QuantumCircuit:
        """Build and return a fresh, unbound variational circuit."""
        return self.build_circuit()


class BasicAnsatz(Ansatz):
    """Configurable TwoLocal ansatz with a contiguous ancilla/system layout.

    Each default layer applies RY rotations followed by a linear CNOT ladder.
    Ancillas occupy the lower-index qubits when used by the Gibbs workflows.
    """

    def __init__(
        self,
        lattice: Lattice,
        num_layers: int = 1,
        num_ancillas: int | None = None,
        rotation_blocks: str | list[str] = "ry",
        entanglement_blocks: str | list[str] = "cx",
        entanglement: str = "linear",
    ):
        """Store the layer, gate, and entanglement settings.

        Args:
            lattice: The lattice on which the model is defined.
            num_layers: Number of repeated rotation/entanglement layers.
            num_ancillas: Number of ancillas; None uses the number of system
                sites when the circuit is built.
            rotation_blocks: Qiskit single-qubit rotation gate names.
            entanglement_blocks: Qiskit entangling gate names.
            entanglement: Qiskit TwoLocal entanglement strategy.
        """
        super().__init__(lattice)
        self.num_layers = num_layers
        self.num_ancillas = num_ancillas
        self.rotation_blocks = rotation_blocks
        self.entanglement_blocks = entanglement_blocks
        self.entanglement = entanglement

    def __str__(self) -> str:
        """Return the human-readable ansatz name."""
        return "Basic Ansatz"

    def build_circuit(self) -> QuantumCircuit:
        """Build a decomposed TwoLocal circuit without a final rotation layer.

        If unset, ``num_ancillas`` is initialized to the number of system sites.
        """
        if self.num_ancillas is None:
            self.num_ancillas = self.lattice.num_nodes
        num_qubits = self.lattice.num_nodes + self.num_ancillas
        qc = TwoLocal(
            num_qubits,
            rotation_blocks=self.rotation_blocks,
            entanglement_blocks=self.entanglement_blocks,
            entanglement=self.entanglement,
            reps=self.num_layers,
            insert_barriers=False,
            skip_final_rotation_layer=True,
        )
        return qc.decompose()


class InterleavedPairAnsatz(Ansatz):
    """Pair-explicit HEA with ancillas interleaved among system sites.

    Logical system sites use the lattice's existing row-major numbering.
    For ``N`` system sites and ``N_a`` ancillas, zero-based ancilla ``k`` is
    paired with system site ``floor(k * N / N_a)`` and placed immediately
    before it in the circuit/MPS ordering. Layout properties are computed
    from the current configuration rather than cached.
    """

    def __init__(
        self,
        lattice: Lattice,
        num_layers: int = 1,
        num_ancillas: int | None = None,
    ):
        """Initialize the pair-explicit layout and validate its dimensions.

        Args:
            lattice: Lattice defining the ordered logical system sites.
            num_layers: Positive number of circuit layers.
            num_ancillas: Number of ancillas, from zero through the number of
                system sites; None pairs one ancilla with every system site.
        """
        super().__init__(lattice)
        self.num_layers = self._validated_integer(
            "num_layers", num_layers, minimum=1
        )
        if num_ancillas is None:
            num_ancillas = lattice.num_nodes
        self.num_ancillas = self._validated_num_ancillas(num_ancillas)

    @staticmethod
    def _validated_integer(name: str, value: object, *, minimum: int) -> int:
        """Return an integer at least ``minimum``, rejecting booleans."""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        result = int(value)
        if result < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return result

    def _validated_num_ancillas(self, value: object) -> int:
        """Validate that the ancilla count lies between zero and system size."""
        result = self._validated_integer(
            "num_ancillas", value, minimum=0
        )
        if result > self.lattice.num_nodes:
            raise ValueError(
                "num_ancillas must satisfy "
                f"0 <= num_ancillas <= {self.lattice.num_nodes}"
            )
        return result

    def _current_dimensions(self) -> tuple[int, int, int]:
        """Return validated ``(N, N_a, L)`` for the current setting."""

        num_system = self.lattice.num_nodes
        num_ancillas = self._validated_num_ancillas(self.num_ancillas)
        num_layers = self._validated_integer(
            "num_layers", self.num_layers, minimum=1
        )
        return num_system, num_ancillas, num_layers

    def _layout(
        self,
    ) -> tuple[
        tuple[int, ...],
        tuple[str, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        """Compute pairing, labels, ancilla positions, and system positions."""

        num_system, num_ancillas, _ = self._current_dimensions()
        paired_system_indices = (
            tuple(
                (ancilla_index * num_system) // num_ancillas
                for ancilla_index in range(num_ancillas)
            )
            if num_ancillas
            else ()
        )
        ancilla_for_system = {
            system_index: ancilla_index
            for ancilla_index, system_index in enumerate(
                paired_system_indices
            )
        }

        site_labels: list[str] = []
        ancilla_positions = [-1] * num_ancillas
        system_positions = [-1] * num_system
        for system_index in range(num_system):
            ancilla_index = ancilla_for_system.get(system_index)
            if ancilla_index is not None:
                ancilla_positions[ancilla_index] = len(site_labels)
                site_labels.append(f"A{ancilla_index + 1}")
            system_positions[system_index] = len(site_labels)
            site_labels.append(f"S{system_index + 1}")

        return (
            paired_system_indices,
            tuple(site_labels),
            tuple(ancilla_positions),
            tuple(system_positions),
        )

    @property
    def site_labels(self) -> tuple[str, ...]:
        """Return one-based A/S labels in circuit and MPS site order."""

        return self._layout()[1]

    @property
    def system_positions(self) -> tuple[int, ...]:
        """Return zero-based circuit/MPS positions in logical system order."""

        return self._layout()[3]

    @property
    def pair_positions(self) -> tuple[tuple[int, int], ...]:
        """Return circuit/MPS control-target positions for the pair CNOTs."""

        paired, _, ancillas, systems = self._layout()
        return tuple(
            (ancillas[ancilla_index], systems[system_index])
            for ancilla_index, system_index in enumerate(paired)
        )

    def __str__(self) -> str:
        """Return the human-readable ansatz name."""
        return "Interleaved Pair Ansatz"

    def build_circuit(self) -> QuantumCircuit:
        """Build rotations, pair CNOTs, then the system CNOT staircase."""

        num_system, _, num_layers = self._current_dimensions()
        site_labels = self.site_labels
        pair_positions = self.pair_positions
        system_positions = self.system_positions
        num_qubits = len(site_labels)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector("theta", num_qubits * num_layers)

        for layer in range(num_layers):
            offset = layer * num_qubits
            for position in range(num_qubits):
                qc.ry(params[offset + position], position)
            for ancilla_position, system_position in pair_positions:
                qc.cx(ancilla_position, system_position)
            for system_index in range(num_system - 1):
                qc.cx(
                    system_positions[system_index],
                    system_positions[system_index + 1],
                )
        return qc


class TFDAnsatz(Ansatz):
    """QAOA-inspired thermofield-double ansatz on two copies of a lattice.

    The circuit starts from paired Bell states and uses four shared rotation
    parameters per layer; each copy has one qubit per logical system site.
    """

    def __init__(self, lattice: Lattice, model: str, num_layers: int = 1):
        """Store the model and use one ancilla per system site.

        Args:
            lattice: The lattice on which the model is defined.
            model: Circuit model: "classical_ising", "transverse_ising", or
                "xxz". This constructor stores the string without validation.
            num_layers: Number of repeated four-parameter layers.
        """
        super().__init__(lattice)
        self.model = model
        self.num_layers = num_layers
        self.num_ancillas = lattice.num_nodes

    def __str__(self) -> str:
        """Return the human-readable ansatz name."""
        return "TFD Ansatz"

    def build_circuit(self) -> QuantumCircuit:
        """Build Bell pairs, within-copy rotations, and paired XX/ZZ rotations.

        The lower-index copy occupies qubits ``0..N-1`` and the second copy
        occupies ``N..2*N-1``. Qiskit rotation parameters are angles in radians.
        """
        qc = QuantumCircuit(self.lattice.num_nodes * 2)
        params = ParameterVector("θ", 4 * self.num_layers)
        # Create the maximally entangled state between the two copies of quantum system A and B
        for i in range(self.lattice.num_nodes):
            qc.h(i)
            qc.cx(i, i + self.lattice.num_nodes)

        for d in range(self.num_layers):
            # H_{A} & H_{B}
            for i, j in self.lattice.interaction_map:
                if self.model == "classical_ising" or self.model == "transverse_ising":
                    qc.rzz(params[4 * d], i, j)
                    qc.rzz(
                        params[4 * d],
                        i + self.lattice.num_nodes,
                        j + self.lattice.num_nodes,
                    )
                elif self.model == "xxz":
                    qc.rxx(params[4 * d], i, j)
                    qc.rxx(
                        params[4 * d],
                        i + self.lattice.num_nodes,
                        j + self.lattice.num_nodes,
                    )
                    qc.ryy(params[4 * d], i, j)
                    qc.ryy(
                        params[4 * d],
                        i + self.lattice.num_nodes,
                        j + self.lattice.num_nodes,
                    )
                    qc.rzz(params[4 * d + 1], i, j)
                    qc.rzz(
                        params[4 * d + 1],
                        i + self.lattice.num_nodes,
                        j + self.lattice.num_nodes,
                    )

            for i in sorted(self.lattice.onsite_field_map):
                # remaining H_{A} & H_{B} for the Ising model
                if self.model == "classical_ising":
                    qc.rz(params[4 * d + 1], i)
                    qc.rz(params[4 * d + 1], i + self.lattice.num_nodes)
                if self.model == "transverse_ising":
                    qc.rx(params[4 * d + 1], i)
                    qc.rx(params[4 * d + 1], i + self.lattice.num_nodes)
                # H_{AB} coupling the A and B systems
                qc.rxx(params[4 * d + 2], i, i + self.lattice.num_nodes)
                qc.rzz(params[4 * d + 3], i, i + self.lattice.num_nodes)

        return qc

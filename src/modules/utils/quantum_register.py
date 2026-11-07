"""
Density-matrix register with labeled tensor factors and operation history.

The communication circuit stays QubitCircuit(1). The state may also contain
Eve's probes and explicitly flagged erasure outputs. Channel maps may change
factor dimensions, but must preserve the number of named tensor factors.

Once a subsystem is an erasure-flagged qutrit, later qubit gates and qubit
channels act only on its {|0>, |1>} block and leave |e> = |2> unchanged: a
lost photon stays lost and nothing acts on it.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from qutip import Qobj, ket2dm, tensor, qeye, sigmax, sigmaz
from qutip_qip.circuit import QubitCircuit

ERASURE = 2 # Measurement outcome for a lost photon: level |e> = |2> of an erasure qutrit.
UNKNOWN = -1 # Placeholder for a bit that Eve could not determine.

_QUBIT_INTO_QUTRIT = Qobj([[1, 0], [0, 1], [0, 0]], dims=[[3], [2]]) # The isometric embedding a qubit into the {|0>, |1>} block of an erasure qutrit. 
_ERASED = Qobj(np.diag([0, 0, 1])) # Projector |e><e| onto the erasure level of a qutrit.

@dataclass(frozen=True)
class RegisterEvent:
    """
    One recorded operation; the history of these is used to draw the circuit.
    """

    kind: str       # "prepare", "ancilla", "gate", "noise", "measure", "note" or "region"
    name: str       # label shown in the drawing
    targets: tuple  # register labels the operation acted on
    input_dims: tuple = ()
    output_dims: tuple = ()
    outcome: int | None = None  # measurement result (measurements only)
    erased: bool = False        # True when a measurement found the photon lost
    controls: int = 0  # The first `controls` entries of targets are control wires.
    basis: str | float | None = None  # measurement basis: "Z", "X", an angle, or None for projectors


class QuantumRegister:
    ERASURE = ERASURE

    def __init__(self):
        self.circuit = QubitCircuit(1)  # communication circuit that transmit() runs on the signal
        self.state = None               # density matrix (Qobj), set by load()
        self.labels = []                # one name per tensor factor, in tensor order
        self.signal = "B"               # label of the qubit that travels through the channel
        self.erasure_levels = {}        # label -> ERASURE for erasure-flagged qutrits
        self.history = []               # RegisterEvent records, used by display()

    @property
    def dims(self):
        """
        Dimension of each tensor factor.
        """

        return [] if self.state is None else list(self.state.dims[0])

    def _targets(self, targets):
        """
        Accept one label or a list of labels; check they exist and are distinct.
        """

        if self.state is None:
            raise ValueError("Load a state first")
        targets = [targets] if isinstance(targets, str) else list(targets)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("Targets must be nonempty and distinct")
        if any(target not in self.labels for target in targets):
            raise ValueError(f"Unknown target; available labels: {self.labels}")
        return targets

    def load(self, state, labels=("B",), signal="B", erasure_levels=None, name="Input state"):
        """
        Replace the register contents with a new state and start a new history.
        """

        if not isinstance(state, Qobj):
            raise TypeError("state must be a QuTiP Qobj")

        rho = ket2dm(state) if state.isket else state.copy() # A ket |psi> is stored as the density matrix |psi><psi|.
        labels = list(labels)

        # Check that rho is a valid density matrix.
        if not rho.isoper or rho.dims[0] != rho.dims[1] or not rho.isherm:
            raise ValueError("state must be a ket or Hermitian density operator")
        if not np.isfinite(rho.full()).all() or not np.isclose(rho.tr(), 1, atol=1e-9, rtol=0):
            raise ValueError("state must be finite and normalized")
        if np.linalg.eigvalsh(rho.full()).min() < -1e-9:
            raise ValueError("state must be positive semidefinite")

        # Check the labels: one unique string per tensor factor, including the signal.
        if len(labels) != len(rho.dims[0]) or len(set(labels)) != len(labels):
            raise ValueError("Supply one unique label per tensor factor")
        if any(not isinstance(label, str) for label in labels) or signal not in labels:
            raise ValueError("Labels must be strings and include signal")

        # Erasure flags are only allowed on three-level (qutrit) factors.
        flags = dict(erasure_levels or {})
        for label, level in flags.items():
            if label not in labels or rho.dims[0][labels.index(label)] != 3 or level != ERASURE:
                raise ValueError("Erasure flags use level 2 of a three-level subsystem")

        # Dividing by the trace removes the small normalization error tolerated above.
        self.state = rho / rho.tr()
        self.labels = labels
        self.signal = signal
        self.erasure_levels = flags
        self.history = [RegisterEvent("prepare", name, tuple(labels), (), tuple(self.dims))]
        return self

    def copy(self):
        return deepcopy(self)

    def _expand(self, operator, targets):
        """
        Embed square or rectangular operators, respecting target order.

        The result acts as `operator` on the targets and as the identity on
        every other factor of the register.
        """
        targets = self._targets(targets)
        indices = [self.labels.index(t) for t in targets]

        # The operator's input dimensions must match the target factors.
        required = [self.dims[i] for i in indices]
        if not isinstance(operator, Qobj) or operator.dims[1] != required:
            raise ValueError(f"Operator input dims must be {required} for {targets}")
        if len(operator.dims[0]) != len(indices):
            raise ValueError("A map must retain one output tensor factor per target")

        # Build operator ⊗ I ⊗ ... ⊗ I with the targets first and the other
        # factors after them, then move every factor back to its register position.
        rest = [i for i in range(len(self.labels)) if i not in indices]
        order = indices + rest
        expanded = tensor(operator, *(qeye(self.dims[i]) for i in rest))

        # permutation[i] is the position in `expanded` of register factor i.
        permutation = [order.index(i) for i in range(len(order))]
        if permutation == list(range(len(order))):
            return expanded  # Already in register order.
        if expanded.dims[0] == expanded.dims[1]:
            return expanded.permute(permutation)

        # Qobj.permute rejects rectangular operators; permute their tensor axes.
        # Reshape the matrix to one axis per output factor followed by one axis
        # per input factor, reorder both groups the same way, and flatten again.
        n = len(order)
        out_dims, in_dims = expanded.dims
        array = expanded.full().reshape(out_dims + in_dims)
        array = array.transpose(permutation + [n + i for i in permutation])
        new_out = [out_dims[i] for i in permutation]
        new_in = [in_dims[i] for i in permutation]
        return Qobj(array.reshape(int(np.prod(new_out)), int(np.prod(new_in))), dims=[new_out, new_in])

    def _is_flagged_qutrit(self, target):
        """
        True if the target is an erasure-flagged qutrit (a signal that may be lost).
        """

        return target in self.erasure_levels and self.dims[self.labels.index(target)] == 3

    def _lift_unitary(self, operator, targets):
        """
        Extend a qubit unitary to erasure-flagged qutrit targets.

        It acts as before when every flagged target is in its qubit block and
        as the identity whenever any flagged target is erased.
        """
        flagged = [self._is_flagged_qutrit(t) for t in targets]

        # Only a qubit operator that touches a flagged target needs lifting.
        if not any(flagged) or operator.dims[1] != [2]*len(targets):
            return operator

        # W embeds every target's qubit space into its register dimension: the
        # qutrit embedding for flagged targets and the identity for the others.
        W = tensor(*(_QUBIT_INTO_QUTRIT if f else qeye(2) for f in flagged))
        identity = qeye([3 if f else 2 for f in flagged])

        # W U W^dag acts on the "nothing erased" subspace, and I - W W^dag is the
        # identity on every state in which at least one flagged target is erased.
        return W*operator*W.dag() + identity - W*W.dag()

    def _lift_kraus(self, operators, targets):
        """
        Extend a single-target qubit channel to an erasure-flagged qutrit.
        """
        if len(targets) != 1:
            raise ValueError("Multi-target qubit channels on erasure-flagged qutrits are not supported")
        out = operators[0].dims[0]
        V = _QUBIT_INTO_QUTRIT
        if out == [2]:
            # A qubit-to-qubit channel acts inside the qubit block: V K V^dag.
            lifted = [V*K*V.dag() for K in operators]
        elif out == [3]:  # A further erasure: its |e> output reuses level 2.
            lifted = [K*V.dag() for K in operators]
        else:
            raise ValueError("A lifted qubit channel must output a qubit or an erasure qutrit")

        # The extra Kraus operator |e><e| keeps an already-lost photon lost.
        return lifted + [_ERASED.copy()]

    def postselect(self, target, projector, name="Postselect"):
        """Project onto a subspace and renormalize without sampling; return its probability."""
        targets = self._targets(target)
        P = self._expand(projector, targets)

        # The trace of the unnormalized state P rho P^dag is the outcome probability.
        branch = P*self.state*P.dag()
        probability = float(np.real(branch.tr()))
        if probability <= 1e-12:
            raise ValueError("The postselected outcome has zero probability")

        self.state = branch / probability
        self.history.append(RegisterEvent("note", name, tuple(targets)))
        return probability

    def add_ancilla(self, name, state, preparation="Init"):
        """Append a new subsystem (for example one of Eve's probes) as the last tensor factor."""
        if self.state is None:
            raise ValueError("Load the signal before adding an ancilla")
        if name in self.labels or not isinstance(name, str):
            raise ValueError("Ancilla labels must be unique strings")

        # Loading into a scratch register reuses load()'s validation of the state.
        check = QuantumRegister().load(state, labels=(name,), signal=name)
        self.state = tensor(self.state, check.state)
        self.labels.append(name)
        self.history.append(RegisterEvent("ancilla", preparation, (name,), (), tuple(check.dims)))
        return self

    @staticmethod
    def _check_unitary(operator, what="operator"):
        if not isinstance(operator, Qobj):
            raise TypeError(f"{what} must be a QuTiP Qobj")
        if operator.dims[0] != operator.dims[1]:
            raise ValueError("A unitary must have equal input/output dimensions")

        # A unitary U satisfies U^dag U = I.
        matrix = operator.full()
        if not np.isfinite(matrix).all() or not np.allclose(matrix.conj().T @ matrix, np.eye(len(matrix)), rtol=0, atol=1e-9):
            raise ValueError(f"{what} is not unitary")

    def apply_unitary(self, operator, targets="B", name="U"):
        targets = self._targets(targets)
        self._check_unitary(operator)

        # Lift onto any erasure-flagged qutrits, then embed into the full register.
        U = self._expand(self._lift_unitary(operator, targets), targets)
        dims = tuple(self.dims[self.labels.index(t)] for t in targets)

        # Unitary evolution of a density matrix: rho -> U rho U^dag.
        self.state = U*self.state*U.dag()
        self.history.append(RegisterEvent("gate", name, tuple(targets), dims, dims))
        return self

    def apply_circuit(self, circuit, targets):
        """
        Run a unitary QubitCircuit and retain each gate for display.

        targets[i] is the register label used for circuit wire i.
        """
        targets = self._targets(targets)
        if len(targets) != circuit.N:
            raise ValueError("Provide one register label per circuit wire")
        if any(getattr(g, "classical_controls", None) for g in circuit.gates):
            raise ValueError("Use register.measure for measurements and feedback")

        steps, events = [], []
        # Each gate's own unitary (controls first, then targets), lifted on its own
        # wires, so a lost qubit blocks only the gates that touch it.
        for gate, unitary in zip(circuit.gates, circuit.propagators(expand=False)):
            controls = list(gate.controls or [])
            local = controls + list(gate.targets or [])
            # Register labels of the wires this gate touches (all wires if it names none).
            involved = [targets[i] for i in local] if local else targets

            # Display name, e.g. "RX(0.3)", or the gate's arg_label when it has one.
            name = gate.name
            if gate.arg_value is not None:
                # Gates such as QASMU and MS take a list of parameters.
                values = ", ".join(f"{v:.3g}" for v in np.ravel(gate.arg_value))
                shown = gate.arg_label if gate.arg_label is not None else values
                name = f"{name}({shown})"

            dims = tuple(self.dims[self.labels.index(t)] for t in involved)

            # A global phase may come back as a plain number: no effect on a density matrix.
            if isinstance(unitary, Qobj) or gate.name != "GLOBALPHASE":
                if gate.name in circuit.user_gates:  # Library gates are unitary by construction.
                    self._check_unitary(unitary, f"Gate {gate.name}")
                steps.append(self._expand(self._lift_unitary(unitary, involved), involved))

            # Controls are stored first, then targets; record how many are controls.
            events.append(RegisterEvent("gate", name, tuple(involved), dims, dims,
                                        controls=len(controls)))

        # All gates are checked before any is applied, so a failure leaves the state unchanged.
        for U in steps:
            self.state = U*self.state*U.dag()
        self.history.extend(events)
        return self

    def apply_kraus(self, operators, targets="B", name="Noise", erasure_levels=None):
        """
        Apply the channel rho -> sum_K K rho K^dag to the targets.

        erasure_levels marks channel outputs that are erasure-flagged qutrits.
        """
        targets = self._targets(targets)
        operators = list(operators)
        if not operators:
            raise ValueError("A channel needs at least one Kraus operator")

        if (any(self._is_flagged_qutrit(t) for t in targets)
                and all(K.dims[1] == [2]*len(targets) for K in operators)):
            # A qubit channel on a possibly-lost signal: act on the qubit block only.
            operators = self._lift_kraus(operators, targets)
            # The output stays an erasure-flagged qutrit.
            erasure_levels = dict(erasure_levels or {})
            erasure_levels[targets[0]] = ERASURE

        # Every Kraus operator must map the targets' current dims to the same output dims.
        before = tuple(self.dims[self.labels.index(t)] for t in targets)
        after = tuple(operators[0].dims[0])
        if any(K.dims != [list(after), list(before)] for K in operators):
            raise ValueError("All Kraus operators must share the target input and output dims")

        # Trace preservation: sum_K K^dag K must equal the identity.
        completeness = sum((K.dag()*K for K in operators), 0*qeye(list(before)))
        if (completeness - qeye(list(before))).norm() > 1e-9:
            raise ValueError("Kraus operators must satisfy sum(K.dag()*K) = I")

        # A factor whose dimension changes loses its old erasure flag; then the
        # flags requested for this channel's outputs are added.
        flags = dict(self.erasure_levels)
        for target, old_dim, new_dim in zip(targets, before, after):
            if old_dim != new_dim:
                flags.pop(target, None)
        for target, level in dict(erasure_levels or {}).items():
            if target not in targets or after[targets.index(target)] != 3 or level != ERASURE:
                raise ValueError("Erasure flags use level 2 of a three-level output")
            flags[target] = level

        # rho -> sum_K K rho K^dag, with each K embedded in the full register.
        result = None
        for K in operators:
            expanded = self._expand(K, targets)
            term = expanded*self.state*expanded.dag()
            result = term if result is None else result + term

        self.state = result
        self.erasure_levels = flags
        self.history.append(RegisterEvent("noise", name, tuple(targets), before, after))
        return self

    def measurement_projectors(self, target="B", basis="Z"):
        """
        Projectors for measuring one target in "Z", "X", or at an angle in the x-z plane.

        A list of projectors is returned unchanged. On an erasure-flagged qutrit
        the qubit projectors are embedded and a projector onto |e> is added.
        """
        self._targets(target)  # Only validates the label.
        dim = self.dims[self.labels.index(target)]

        # Explicit projectors (anything that is not a string or a number) pass through.
        if not isinstance(basis, str) and not np.isscalar(basis):
            return list(basis)
        if dim != 2 and target not in self.erasure_levels:
            raise ValueError("For an unflagged qudit, provide explicit measurement projectors")

        # The observable's +1 and -1 eigenspaces are outcomes 0 and 1.
        if isinstance(basis, str):
            if basis not in ("X", "Z"):
                raise ValueError("basis must be X, Z, a real angle, or projectors")
            observable = sigmax() if basis == "X" else sigmaz()
        else:
            angle = float(basis)
            if not np.isfinite(angle):
                raise ValueError("Measurement angle must be finite")
            # Bloch-sphere direction rotated by `angle` from Z toward X.
            observable = np.cos(angle)*sigmaz() + np.sin(angle)*sigmax()

        # (I + O)/2 and (I - O)/2 project onto the +1 and -1 eigenspaces of O.
        projectors = [(qeye(2) + observable)/2, (qeye(2) - observable)/2]
        if dim == 3:
            # Erasure qutrit: embed the qubit projectors and add outcome 2, |e><e|.
            V = _QUBIT_INTO_QUTRIT
            projectors = [V*P*V.dag() for P in projectors]
            projectors.append(_ERASED.copy())
        return projectors

    def measure(self, target="B", basis="Z", rng=None, name=None):
        """
        Projective measurement; returns the outcome index.

        One target accepts "X", "Z", an angle, or projectors. Several targets
        (a joint measurement) require explicit projectors on their joint space.
        """
        
        targets = self._targets(target)
        dims = [self.dims[self.labels.index(t)] for t in targets]

        if len(targets) == 1:
            projectors = self.measurement_projectors(targets[0], basis)
        elif isinstance(basis, str) or np.isscalar(basis):
            raise ValueError("A joint measurement on several targets needs explicit projectors")
        else:
            projectors = list(basis)

        # Each projector must have the targets' dims and satisfy P = P^dag = P^2,
        # and together the projectors must sum to the identity.
        if not projectors or any(P.dims != [dims, dims] or not P.isherm or (P*P - P).norm() > 1e-9 for P in projectors):
            raise ValueError("Supply projectors with the target's dimensions")
        if (sum(projectors, 0*qeye(dims)) - qeye(dims)).norm() > 1e-9:
            raise ValueError("Measurement projectors must sum to the identity")

        # Born rule: the probability of outcome k is Tr(P_k rho).
        expanded = [self._expand(P, targets) for P in projectors]
        probs = np.array([float(np.real((P*self.state).tr())) for P in expanded])
        if probs.min() < -1e-9 or not np.isfinite(probs).all():
            raise ValueError("Invalid measurement probabilities")

        # Clip tiny negative roundoff, then renormalize before sampling.
        probs = np.maximum(probs, 0)
        if probs.sum() <= 0:
            raise ValueError("Measurement has zero total probability")
        probs /= probs.sum()

        rng = np.random.default_rng() if rng is None else rng
        outcome = int(rng.choice(len(projectors), p=probs))

        # Collapse onto the sampled outcome: rho -> P rho P^dag / Tr(P rho P^dag).
        branch = expanded[outcome]*self.state*expanded[outcome].dag()
        self.state = branch / branch.tr()

        # Label for the drawing: the basis name, the angle, "custom", or `name` if given.
        if isinstance(basis, str):
            label = basis
        elif np.isscalar(basis):
            label = f"{basis:.3g} rad"
        else:
            label = "custom"
        if name is not None:
            label = name

        # The exact basis lets the drawing rotate into it (and resend from it after an intercept).
        if isinstance(basis, str):
            drawn_basis = str(basis)
        elif np.isscalar(basis):
            drawn_basis = float(basis)
        else:
            drawn_basis = None

        # The photon was lost if the outcome is the erasure projector |e><e|.
        erased = len(targets) == 1 and targets[0] in self.erasure_levels and (projectors[outcome] - _ERASED).norm() < 1e-9

        self.history.append(RegisterEvent("measure", label, tuple(targets), tuple(dims), tuple(dims), outcome, bool(erased), basis=drawn_basis))
        return outcome

    def reduced(self, targets="B"):
        """
        Reduced density matrix of the targets, with factors in the order given.
        """

        targets = self._targets(targets)
        indices = [self.labels.index(t) for t in targets]

        # ptrace keeps factors in register order; permute them into the requested order.
        ordered = sorted(indices)
        return self.state.ptrace(ordered).permute([ordered.index(i) for i in indices])

    def transmit(self, attack=None, channel=None, *, mark_regions=False):
        """
        Communication circuit -> optional Alice-Eve channel on the signal -> attack step(s).
        """

        if self.circuit.gates:
            if mark_regions:
                self.mark_region("Transmission")
            self.apply_circuit(self.circuit, [self.signal])

        if channel is not None:
            if mark_regions:
                self.mark_region("Quantum channels")
            channel.apply(self, self.signal)

        # The attack may be None, one callable, or a list/tuple applied in order.
        if attack is None:
            steps = []
        elif isinstance(attack, (list, tuple)):
            steps = attack
        else:
            steps = [attack]

        for step in steps:
            if mark_regions:
                self.mark_region("Attack")
            step(self)
        return self

    def mark_region(self, name):
        """
        Start a new titled region in the drawing, drawn as a barrier (see circuit_display).
        """

        self.history.append(RegisterEvent("region", name, ()))

    def annotate(self, name, targets="B"):
        """
        Record a note; notes are not drawn as gates but the display code reads some of them.
        """

        targets = self._targets(targets)
        self.history.append(RegisterEvent("note", name, tuple(targets)))
        return self

    def display(self, theme="iqp", dpi=300, save_path=None, show=True, title=None):
        """
        Draw the operation history as a Qiskit circuit and return its Matplotlib Figure.

        QuTiP still computes everything; Qiskit only draws. theme is a Qiskit
        style ("iqp", "iqp-dark", "clifford", "textbook", "bw" or a style dict);
        QuTiP's theme names ("qutip", "light", "dark", "modern") still work.
        """

        from .circuit_display import draw_register
        return draw_register(self, theme=theme, dpi=dpi, save_path=save_path, show=show, title=title)

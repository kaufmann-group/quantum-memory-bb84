"""Draw a register's recorded history as a Qiskit circuit diagram.

QuTiP performs every calculation; Qiskit is used only to draw. history_circuit()
translates the QuantumRegister history into a display-only QuantumCircuit:

* every register label (B, A, E, F, ...) is a one-qubit wire, in tensor order;
* every classical result (Alice's, Bob's, Eve's reads or her combined guess)
  is a ClassicalRegister, and its wire label shows the recorded value;
* every region marker (Attack, Measurement, ...) is a labeled barrier;
* a measured qubit that is used again afterwards, such as a photon that Eve
  intercepts and forwards, is re-prepared by classical feed-forward. A
  destructive measurement leaves nothing to send on, so the qubit is reset to
  |0>, an X gate inside if_test(c == 1) loads the measured bit, and the
  rotation into the measurement basis turns |c> into the eigenstate found.

Noise channels, other state preparations and joint measurements are labeled
boxes, so the circuit is a schematic of the recorded run, not an executable
replacement for the register's density-matrix evolution.

Requires qiskit with its visualization extras: pip install "qiskit[visualization]".
"""

import re
from collections import Counter
from inspect import signature
from math import isclose
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit import QuantumRegister as QiskitRegister
from qiskit.circuit import ControlFlowOp, ControlledGate, Gate, Instruction

# Title of the events recorded before the first region marker.
_FIRST_REGION = "Alice bit preparation"

# QuTiP's theme names map to the closest Qiskit styles; anything else goes to Qiskit unchanged.
_THEMES = {"qutip": "iqp", "light": "clifford", "modern": "textbook", "dark": "iqp-dark"}

# Inches per Qiskit drawing unit. At this size Qiskit draws its fonts at their nominal size.
_INCHES_PER_UNIT = 0.8361111**2

# Plain-text state names in the history -> LaTeX labels for preparation boxes.
_KET_LABELS = {
    "|1>": r"$|1\rangle$",
    "|+>": r"$|+\rangle$",
    "|->": r"$|-\rangle$",
    "|e0>": r"$|e_0\rangle$",
    "|f0>": r"$|f_0\rangle$",
}

# QuTiP gate name -> (QuantumCircuit method, number of wires). Recorded wires list
# controls first, then targets, which is also Qiskit's argument order.
_LIBRARY_GATES = {
    "X": ("x", 1), "Y": ("y", 1), "Z": ("z", 1), "SNOT": ("h", 1),
    "S": ("s", 1), "T": ("t", 1), "SQRTNOT": ("sx", 1),
    "CNOT": ("cx", 2), "CX": ("cx", 2), "CY": ("cy", 2), "CZ": ("cz", 2), "CSIGN": ("cz", 2),
    "CS": ("cs", 2), "SWAP": ("swap", 2), "ISWAP": ("iswap", 2),
    "TOFFOLI": ("ccx", 3), "FREDKIN": ("cswap", 3),
}

# Single-qubit rotations recorded as e.g. "RX(0.3)".
_ROTATIONS = {"RX": "rx", "RY": "ry", "RZ": "rz", "PHASEGATE": "p"}


def _parties(register):
    """Register label -> legitimate party that measures it, in display order."""
    first = register.history[0].name
    if first == "Alice |0>":    # BB84: Bob measures the signal.
        return {register.signal: "Bob"}
    if first == "Source |00>":  # E91: Alice and Bob each measure one qubit of the pair.
        return {"A": "Alice", "B": "Bob"}
    return {}


def _tex(owner):
    """Upright LaTeX for an owner such as "Eve,guess"."""
    parts = [part.replace("_", r"\_") for part in owner.split(",")]
    return ",".join(r"\mathrm{" + part + "}" for part in parts)


def _classical_registers(register, events):
    """One ClassicalRegister per classical result: the parties' first, then Eve's in time order.

    Returns (registers, writes, labels): writes maps id(measurement event) to
    (register, bit), and labels maps each register name to (owner, value) LaTeX.
    """
    measurements = [event for event in events if event.kind == "measure"]
    order = {id(event): i for i, event in enumerate(events)}
    parties = _parties(register)

    # A party's result is its last measurement on its own qubit. An earlier one
    # on the same qubit (an intercept) is Eve's.
    outputs, claimed = [], set()
    for label, owner in parties.items():
        read = next((e for e in reversed(measurements) if e.targets == (label,)), None)
        if read is not None:
            outputs.append((owner, read.outcome, [read], ()))
            claimed.add(id(read))

    # Eve's final guesses were recorded as notes such as "Eve guess: 1" on her probes.
    guesses = {e.targets: int(e.name.split(": ")[1]) for e in events
               if e.kind == "note" and e.name.startswith("Eve guess: ")}

    # Each of Eve's outputs is (time, owner, value, contributing reads, probe labels).
    eve = []
    # A multi-probe attack (FGGNP) combines several probe reads into one guess.
    for note in events:
        if note.kind == "note" and note.name == "FGGNP probe register" and note.targets in guesses:
            reads = [m for m in measurements
                     if id(m) not in claimed and set(m.targets) <= set(note.targets)]
            if reads:
                eve.append((order[id(reads[0])], "Eve,guess", guesses[note.targets], reads, note.targets))
                claimed.update(id(m) for m in reads)
    # Every other measurement (an intercept or a probe read) has its own result.
    for m in measurements:
        if id(m) not in claimed:
            owner = "Eve" if parties else ",".join(m.targets)
            eve.append((order[id(m)], owner, m.outcome, [m], ()))
    eve.sort(key=lambda output: output[0])
    outputs += [output[1:] for output in eve]

    # An erased photon has no bit. Its detector box still shows the loss.
    outputs = [output for output in outputs
               if not (len(output[2]) == 1 and output[2][0].erased and output[0] != "Eve,guess")]

    # Several results with the same owner (chained attacks) are numbered in time order.
    totals = Counter(output[0] for output in outputs)
    seen, owners = Counter(), []
    for owner, *_ in outputs:
        seen[owner] += 1
        owners.append(f"{owner},{seen[owner]}" if totals[owner] > 1 else owner)

    taken = set(register.labels)  # Qiskit register names must be unique.
    registers, writes, labels = [], {}, {}
    for owner, (_, value, reads, probes) in zip(owners, outputs):
        shown = str(value) if value in (0, 1) else "unknown"
        # The plain-text label as the name: Qiskit sizes the label margin from it,
        # and it stays readable if the LaTeX relabeling does not apply.
        name = f"c_{owner}={shown}"
        while name in taken:
            name += "'"
        taken.add(name)

        # Single-probe reads of a probe group each get their own bit of one register.
        indexed = bool(probes) and all(len(m.targets) == 1 for m in reads)
        creg = ClassicalRegister(len(probes) if indexed else 1, name)
        registers.append(creg)
        labels[name] = (_tex(owner), shown if value in (0, 1) else r"\mathrm{unknown}")
        for m in reads:
            writes[id(m)] = (creg, creg[probes.index(m.targets[0]) if indexed else 0])
    return registers, writes, labels


def _reused_after_measurement(events):
    """For each event: is it a one-qubit measurement whose qubit a later operation uses?"""
    reused = [False] * len(events)
    later = set()  # labels acted on by some later operation
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if event.kind == "measure" and len(event.targets) == 1:
            reused[index] = event.targets[0] in later
        if event.kind in ("gate", "noise", "measure"):
            later.update(event.targets)
    return reused


def _box(circuit, name, label, wires, clbits=()):
    """A labeled box on the wires (and classical bits) it acts on."""
    circuit.append(Instruction(name, len(wires), len(clbits), [], label=label), wires, list(clbits))


def _add_preparation(circuit, event, wires):
    state = str(event.name)
    for prefix in ("Alice ", "Init ", "Source "):
        state = state.removeprefix(prefix)
    if re.fullmatch(r"\|0+>", state) and len(state) - 2 == len(wires):
        # Qiskit draws a reset as a |0> box: one per wire for e.g. "Source |00>".
        for wire in wires:
            circuit.reset(wire)
    else:
        _box(circuit, "prepare", _KET_LABELS.get(state, str(event.name)), wires)


def _add_gate(circuit, event, wires):
    name, controls = str(event.name), event.controls

    library = _LIBRARY_GATES.get(name)
    if library is not None and library[1] == len(wires):
        getattr(circuit, library[0])(*wires)
        return

    rotation = re.fullmatch(r"(RX|RY|RZ|PHASEGATE)\(([^,]+)\)", name)
    if rotation and len(wires) == 1:
        try:
            angle = float(rotation[2])
        except ValueError:
            pass  # A symbolic arg_label such as "pi/2" stays a text box.
        else:
            getattr(circuit, _ROTATIONS[rotation[1]])(angle, wires[0])
            return

    # Any other gate (e.g. PhaseClone, PartialSWAP) is a labeled box, with its control dots.
    box = Gate("custom", len(wires) - controls, [], label=name)
    if controls:
        box = ControlledGate("ccustom", len(wires), [], num_ctrl_qubits=controls, base_gate=box)
    circuit.append(box, wires)


def _rotate(circuit, wire, basis, inverse=False):
    """Map |0>/|1> onto the eigenstates of basis: "Z", "X" or an angle in the x-z plane.

    RY(angle) turns |0> into the +1 eigenstate of cos(angle)*Z + sin(angle)*X.
    inverse=True gives the rotation that precedes a Z-basis meter.
    """
    if basis == "X":
        circuit.h(wire)
    elif basis != "Z":
        circuit.ry(-basis if inverse else basis, wire)


def _resend(circuit, wire, creg, clbit, basis):
    """Re-prepare a measured qubit from its classical outcome (classical feed-forward).

    A classical bit cannot enter a unitary gate directly: the qubit is
    re-initialized, an X conditioned on the bit loads it, and the basis
    rotation turns |bit> into the eigenstate the measurement found.
    """
    circuit.reset(wire)
    # A one-bit register can be the condition itself, which Qiskit draws without a bit index.
    condition = (creg, 1) if creg.size == 1 else (clbit, 1)
    with circuit.if_test(condition):
        circuit.x(wire)
    _rotate(circuit, wire, basis)


def _measurement_basis(event):
    """The recorded basis: "Z", "X", an angle, or None for explicit projectors."""
    basis = getattr(event, "basis", None)
    if basis is None and str(event.name) in ("X", "Z"):
        basis = str(event.name)  # A history recorded before RegisterEvent.basis existed.
    return basis


def _add_measurement(circuit, event, wires, write, resend):
    if write is None or event.erased:
        # No classical bit (a lost photon): show the detector's result in a box.
        value = "erased" if event.erased else str(event.outcome)
        _box(circuit, "readout", f"{event.name} readout: {value}", wires)
        return

    creg, clbit = write
    basis = _measurement_basis(event)
    if basis is None or len(wires) != 1:
        # Explicit projectors (e.g. Helstrom): one joint instrument writing one result.
        _box(circuit, "instrument", str(event.name), wires, [clbit])
        return

    # A measurement in another basis is drawn as the basis rotation followed by a Z meter.
    (wire,) = wires
    _rotate(circuit, wire, basis, inverse=True)
    circuit.measure(wire, clbit)
    if resend:
        _resend(circuit, wire, creg, clbit, basis)


def history_circuit(register, events=None):
    """Build the display-only Qiskit QuantumCircuit of the register's recorded history.

    circuit.metadata holds what the drawing adds on top of Qiskit's own
    rendering: "classical_labels" maps each classical register name to its
    (owner, value) LaTeX, and "first_region" titles the events before the
    first region barrier (None when the history marks no regions).
    """
    if register.state is None or not register.history:
        raise ValueError("Load a state before displaying its history")
    events = register.history if events is None else list(events)

    qubits = {label: QiskitRegister(1, label) for label in register.labels}
    cregs, writes, labels = _classical_registers(register, events)
    circuit = QuantumCircuit(*qubits.values(), *cregs, name="history")
    reused = _reused_after_measurement(events)

    first_region = region = _FIRST_REGION if any(e.kind == "region" for e in events) else None
    pending, started = None, False
    for index, event in enumerate(events):
        # Notes are not drawn; a region marker takes effect at the next drawn operation.
        if event.kind == "note":
            continue
        if event.kind == "region":
            pending = event.name
            continue
        if pending is not None and pending != region:
            # Consecutive markers with the same name (several attacks) share one region.
            if started:
                circuit.barrier(label=pending)
            else:
                first_region = pending
            region = pending
        pending, started = None, True

        wires = [qubits[target][0] for target in event.targets]
        if event.kind == "measure":
            _add_measurement(circuit, event, wires, writes.get(id(event)), reused[index])
        elif event.kind in ("prepare", "ancilla"):
            _add_preparation(circuit, event, wires)
        elif event.kind == "gate":
            _add_gate(circuit, event, wires)
        else:
            # Noise channels (and any other recorded operation) are labeled boxes.
            label = str(event.name)
            if event.input_dims != event.output_dims:
                # Show a dimension change, e.g. "QubitErasure (2→3)".
                change = ",".join(f"{a}→{b}" for a, b in zip(event.input_dims, event.output_dims))
                label += f" ({change})"
            _box(circuit, "channel", label, wires)

    circuit.metadata = {"first_region": first_region, "classical_labels": labels}
    return circuit


def _plain(text):
    """Strip Qiskit's mathtext markup from a wire label: "${c\\_Bob}$" -> "c_Bob"."""
    return re.sub(r"[${}\\]", "", text)


def _font(ax, circuit):
    """Qiskit's base font size and text colour, read from the first qubit's wire label."""
    for text in ax.texts:
        if text.get_ha() == "right" and _plain(text.get_text()) == circuit.qregs[0].name:
            return text.get_fontsize() / 1.25, text.get_color()  # Labels use 1.25 x the font.
    return 13.0, "black"


def _drop_bundle_marks(ax, x, y):
    """Remove the "1/" size marker of a one-bit wire labeled at (x, y) and its "0" bit indices."""
    for text in list(ax.texts):
        tx, ty = text.get_position()
        if (text.get_ha(), text.get_va()) != ("left", "bottom") or not isclose(ty, y + 0.1, abs_tol=1e-6):
            continue
        if ((text.get_text() == "1" and isclose(tx, x + 0.3, abs_tol=1e-6))
                or (text.get_text() == "0" and tx > x + 0.5)):
            text.remove()
    for line in list(ax.lines):
        xs, ys = line.get_data()
        if len(xs) == 2 and np.allclose(xs, (x + 0.4, x + 0.5)) and np.allclose(ys, (y - 0.1, y + 0.1)):
            line.remove()


def _label_classical_wires(ax, circuit, bundled):
    """Write each classical register's owner and recorded value into its wire label."""
    labels = circuit.metadata["classical_labels"]
    wire_texts = {_plain(t.get_text()): t for t in ax.texts if t.get_ha() == "right"}
    for creg in circuit.cregs:
        owner, value = labels[creg.name]
        if bundled or creg.size == 1:
            text = wire_texts.get(creg.name)
            if text is not None:
                text.set_text(f"$c_{{{owner}}}={value}$")
                if bundled and creg.size == 1:
                    # A one-bit register needs neither a bundle size nor bit indices.
                    _drop_bundle_marks(ax, *text.get_position())
        else:
            # Unbundled bits of a probe group are drawn as separate wires.
            for i in range(creg.size):
                text = wire_texts.get(f"{creg.name}_{i}")
                if text is not None:
                    text.set_text(f"$c_{{{owner}}}[{i}]$")


def _wire_extent(ax):
    """x-range of the top quantum wire, which Qiskit draws at y = 0."""
    spans = [line.get_xdata() for line in ax.lines
             if len(line.get_ydata()) == 2 and np.allclose(line.get_ydata(), 0)]
    if not spans:
        return ax.get_xlim()
    return min(min(span) for span in spans), max(max(span) for span in spans)


def _title_regions(ax, circuit, barrier_labels, y, fontsize, color):
    """Center each region's title over its stretch of the wires.

    Qiskit prints a barrier's label on the barrier itself. If the drawing does
    not contain exactly those labels, they are left where Qiskit put them.
    """
    texts = sorted((t for t in ax.texts if t.get_va() == "top" and t.get_text() in barrier_labels),
                   key=lambda t: t.get_position()[0])
    if [t.get_text() for t in texts] != barrier_labels:
        return
    barriers = [t.get_position()[0] for t in texts]
    for text in texts:
        text.remove()

    # Qiskit shortened each labeled barrier's top segment to make room for the label.
    for line in ax.lines:
        xs, ys = line.get_data()
        if (line.get_linestyle() == "--" and len(xs) == 2 and isclose(ys[0], 0.15, abs_tol=1e-6)
                and any(isclose(xs[0], b, abs_tol=1e-6) for b in barriers)):
            line.set_ydata([0.5, ys[1]])
    for patch in ax.patches:
        if (isinstance(patch, Rectangle) and isclose(patch.get_height(), 0.65, abs_tol=1e-6)
                and any(isclose(patch.get_x() + patch.get_width() / 2, b, abs_tol=1e-6) for b in barriers)):
            patch.set_height(1.0)

    start, end = _wire_extent(ax)
    edges = [start, *barriers, end]
    names = [circuit.metadata["first_region"], *barrier_labels]
    for name, left, right in zip(names, edges, edges[1:]):
        ax.text((left + right) / 2, y, name, ha="center", va="bottom", fontsize=fontsize, color=color)


def _draw(circuit, theme, dpi, title):
    """Render the circuit with Qiskit's matplotlib drawer onto a new pyplot Figure."""
    barrier_labels = [inst.operation.label for inst in circuit.data
                      if inst.operation.name == "barrier" and inst.operation.label]
    # Qiskit cannot bundle classical wires once a joint instrument writes to a single bit.
    joint = any(inst.clbits and inst.operation.name != "measure"
                and not isinstance(inst.operation, ControlFlowOp) for inst in circuit.data)
    style = _THEMES.get(theme, theme) if isinstance(theme, str) or theme is None else theme
    options = dict(output="mpl", style=style or "iqp", fold=-1, cregbundle=not joint)
    if "barrier_label_len" in signature(QuantumCircuit.draw).parameters:
        # Newer Qiskit versions shorten barrier labels to 16 characters by default.
        options["barrier_label_len"] = max([16, *map(len, barrier_labels)])

    # Pass 1: Qiskit sizes its own figure; keep the drawing's extent, colours and font.
    sizing = circuit.draw(**options)
    (left, right), (bottom, top) = sizing.axes[0].get_xlim(), sizing.axes[0].get_ylim()
    background = sizing.get_facecolor()
    fontsize, color = _font(sizing.axes[0], circuit)
    plt.close(sizing)

    # Room above the wires for the region titles and the figure title, in drawing units.
    line = fontsize / 72 / _INCHES_PER_UNIT
    region_y = 0.55  # Above the top wire's gate boxes (0.325) and If boxes (0.425).
    if circuit.metadata["first_region"] is not None:
        top = max(top, region_y + 1.25*line)
    title_y = top + 0.25
    if title:
        top = title_y + 1.5*line

    # Pass 2: a pyplot figure of exactly that size, so Qiskit keeps its nominal fonts
    # and show()/close() behave as before (Qiskit closes the figures it creates inline).
    figure = plt.figure(figsize=((right - left)*_INCHES_PER_UNIT, (top - bottom)*_INCHES_PER_UNIT),
                        dpi=dpi, facecolor=background)
    ax = figure.add_axes((0, 0, 1, 1))
    circuit.draw(ax=ax, **options)
    ax.set_ylim(bottom, top)

    _label_classical_wires(ax, circuit, options["cregbundle"])
    if circuit.metadata["first_region"] is not None:
        _title_regions(ax, circuit, barrier_labels, region_y, fontsize, color)
    if title:
        ax.text((left + right) / 2, title_y, title, ha="center", va="bottom",
                fontsize=1.15*fontsize, color=color)
    return figure


def draw_register(register, theme="iqp", dpi=150, save_path=None, show=True, title=None):
    """Draw the register's history with Qiskit and return the Matplotlib Figure.

    theme is a Qiskit style ("iqp", "iqp-dark", "clifford", "textbook", "bw" or
    a style dict); QuTiP's theme names ("qutip", "light", "dark", "modern")
    are mapped to the closest Qiskit style.
    """
    figure = _draw(history_circuit(register), theme, dpi, title)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        # Drop pyplot's reference so batch displays do not accumulate open
        # figures; the returned Figure can still be saved or shown in Jupyter.
        plt.close(figure)
    return figure

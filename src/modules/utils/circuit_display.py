"""Convert recorded operations into a schematic drawn by QuTiP itself.

Channel and measurement boxes annotate the recorded run; this display circuit
is not an executable replacement for the register's density-matrix evolution.
"""

from math import isclose
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate, Measurement

# Plain-text state names in the history -> LaTeX labels for the drawing.
_KET_LABELS = {
    "|0>": r"$|0\rangle$",
    "|1>": r"$|1\rangle$",
    "|e0>": r"$|e_0\rangle$",
    "|f0>": r"$|f_0\rangle$",
    "Source |00>": r"Source $|00\rangle$",
}


def _add_box(diagram, name, wires, n_controls=0, **kwargs):
    """Add a display gate, splitting wires into controls and targets.

    Library names such as CZ or TOFFOLI need their controls passed separately.
    If QuTiP still rejects the arity, draw a generic labeled box instead.
    """
    controls, targets = wires[:n_controls], wires[n_controls:]
    try:
        diagram.add_gate(name, targets=targets, controls=controls or None, **kwargs)
    except (TypeError, ValueError):
        diagram.add_gate(Gate(name=name, targets=targets, controls=controls or None))


def _bb84_readouts(register, groups, bundle_classical):
    """Resolve the owners, classical results and contributing measurements once per drawing.

    Returns (labels, writes): one label per classical wire, and a map from
    id(measurement event) to (classical wire index, bit index or None).
    Only BB84 rounds, whose history starts with "Alice |0>", get classical wires.
    """
    if register.history[0].name != "Alice |0>":
        return [], {}

    measurements = [event for event in register.history if event.kind == "measure"]

    # Bob's measurement is the last one made on the signal qubit.
    bob = next((e for e in reversed(measurements) if e.targets == (register.signal,)), None)

    # Eve's final guesses were recorded as notes such as "Eve guess: 1" on her probes.
    guesses = {}
    for e in register.history:
        if e.kind == "note" and e.name.startswith("Eve guess: "):
            guesses[e.targets] = int(e.name.split(": ")[1])

    # Each output is (owner, classical value, contributing measurements, probe labels).
    outputs = []

    # A multi-probe attack (FGGNP) combines several probe reads into one guess.
    for group in groups:
        reads = [e for e in measurements if set(e.targets).issubset(group)]
        if reads and group in guesses:
            outputs.append((r"\mathrm{Eve},\mathrm{guess}", guesses[group], reads, group))

    # Every remaining measurement except Bob's is one of Eve's and gets its own output.
    grouped = {id(e) for _, _, reads, _ in outputs for e in reads}
    for e in measurements:
        if e is bob or id(e) in grouped:
            continue
        # An intercept acts on B, while a delayed probe read acts on Eve's wire.
        outputs.append((r"\mathrm{Eve}", e.outcome, [e], ()))
    if bob is not None:
        outputs.append((r"\mathrm{Bob}", bob.outcome, [bob], ()))

    labels, writes = [], {}
    for owner, value, reads, probes in outputs:
        # An erased signal has no bit. Its detector box can still show the loss.
        if len(reads) == 1 and reads[0].erased and owner != r"\mathrm{Eve},\mathrm{guess}":
            continue

        bit = str(value) if value in (0, 1) else r"\mathrm{unknown}"
        store = len(labels)  # index of the first classical wire for this output

        # Single-probe reads of a probe group can each get their own bit slot.
        indexed = bool(probes) and all(len(e.targets) == 1 for e in reads)
        if indexed and not bundle_classical:
            # Distinct real storage slots, even in a Z round when F is unused.
            labels.extend(rf"$c_{{\mathrm{{Eve}}}}[{i}]$" for i in range(len(probes)))
        else:
            labels.append(rf"$c_{{{owner}}}={bit}$")

        for e in reads:
            slot = probes.index(e.targets[0]) if indexed else None
            if indexed and not bundle_classical:
                writes[id(e)] = (store + slot, None)  # its own wire
            else:
                writes[id(e)] = (store, slot)         # shared wire, bit index `slot`
    return labels, writes


def history_circuit(register, events=None, *, bundle_classical=False):
    """Build a display-only QubitCircuit from the register's recorded history.

    Keep all quantum wires separate. Only the plotted view bundles classical slots.
    Returns (diagram, wire labels), with the classical labels first.
    """
    if register.state is None or not register.history:
        raise ValueError("Load a state before displaying its history")

    # FGGNP probe groups, used to combine Eve's reads into one guess (BB84 rounds only).
    groups = []
    if register.history[0].name == "Alice |0>":
        groups = [e.targets for e in register.history
                  if e.kind == "note" and e.name == "FGGNP probe register"]

    # QuTiP draws wire 0 lowest, so reversing the labels puts the first factor on top.
    quantum = list(reversed(register.labels))
    indices = {label: i for i, label in enumerate(quantum)}
    classical, writes = _bb84_readouts(register, groups, bundle_classical)
    diagram = QubitCircuit(len(quantum), num_cbits=len(classical))

    if events is None:
        events = register.history

    for event in events:
        # Notes and region markers are not drawn as gates.
        if event.kind in ("note", "region"):
            continue
        wires = [indices[q] for q in event.targets]

        if event.kind == "measure":
            write = writes.get(id(event))
            if write is None:
                # No classical wire for this read (e.g. E91): show the result in a box.
                value = "erased" if event.erased else str(event.outcome)
                diagram.add_gate(f"{event.name} readout: {value}", targets=wires)
                continue

            store, slot = write
            # An X-basis measurement is drawn as a Hadamard followed by a Z meter.
            if event.name == "X":
                diagram.add_gate("SNOT", targets=wires, arg_label="H")
            caption = "M" if event.name in ("X", "Z") else event.name
            if event.erased:
                caption = "erased"

            # One meter per physical read. In the compact rendering, store is
            # the row and slot is the bit index printed at that row's arrowhead.
            diagram.add_measurement(caption, targets=sorted(wires), classical_store=store)
            diagram.gates[-1].index = slot

            # An X intercept resends its measured X eigenstate to Bob.
            if (event.name == "X" and event.targets == (register.signal,)
                    and classical[store].startswith(r"$c_{\mathrm{Eve}}")):
                diagram.add_gate("SNOT", targets=wires, arg_label="H")

        elif event.kind == "gate" and event.name in ("CNOT", "CX"):
            diagram.add_gate("CNOT", controls=wires[:1], targets=wires[1:])
        elif event.kind == "gate" and event.name == "SNOT":
            diagram.add_gate("SNOT", targets=wires, arg_label="H")
        elif event.kind == "gate" and event.controls:
            _add_box(diagram, event.name, wires, event.controls)
        else:
            # Everything else (state preparation, ancillas, other gates, noise) is a labeled box.
            label = event.name.replace("Alice ", "").replace("Init ", "")
            if event.kind in ("prepare", "ancilla"):
                label = _KET_LABELS.get(label, label)
            if event.kind == "noise":
                # Show a dimension change, e.g. "QubitErasure (2→3)".
                if event.input_dims != event.output_dims:
                    change = ",".join(f"{a}→{b}" for a, b in zip(event.input_dims, event.output_dims))
                    label += f" ({change})"
            _add_box(diagram, label, wires)

    return diagram, classical + quantum


def _label_meters(diagram, ax):
    """QuTiP draws meter faces in operation order but omits their captions."""
    meters = [gate for gate in diagram.gates if isinstance(gate, Measurement)]
    faces = [patch for patch in ax.patches if isinstance(patch, Arc)]

    for meter, face in zip(meters, faces):
        x, y = face.center
        color = face.get_edgecolor()

        if meter.index is not None:
            # Match the renderer's wire_sep setting. Its y=0 is classical bit 0.
            ax.annotate(f"[{meter.index}]", (x, 0.5 * meter.classical_store),
                        xytext=(4, 3), textcoords="offset points", fontsize=8,
                        color=color, va="bottom")

        if len(meter.targets) > 1:
            # QuTiP draws one face for a joint instrument. Extend a bracket
            # up to its other input to distinguish it from local probe reads.
            top = 0.5 * (diagram.num_cbits + max(meter.targets))
            ax.plot([x, x, x + 0.08], [y + 0.20, top, top], color=color, lw=1.3)

        if meter.name != "M":
            ax.annotate(meter.name, (x, y + 0.31), xytext=(3, 2),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=8, color=color)


def _draw_regions(register, theme, dpi, title):
    # A panel per recorded phase prevents QuTiP from packing a later
    # independent probe operation into an earlier phase on another wire.

    # Group the drawable events into consecutive (region name, events) panels.
    regions = []
    name = "Alice bit preparation"  # title of the events before the first region marker
    for event in register.history:
        if event.kind == "region":
            name = event.name
        elif event.kind != "note":
            if not regions or regions[-1][0] != name:
                regions.append((name, []))
            regions[-1][1].append(event)

    fig, axes = plt.subplots(1, len(regions), squeeze=False, dpi=dpi)
    widths = []
    for i, ((name, events), ax) in enumerate(zip(regions, axes[0])):
        diagram, labels = history_circuit(register, events, bundle_classical=True)

        # Only the first panel shows the wire labels.
        wire_labels = labels if i == 0 else [""] * len(labels)
        # Keep QuTiP from calling tight_layout on the shared figure; the panels are laid out below.
        with patch("matplotlib.pyplot.tight_layout"):
            diagram.draw("matplotlib", ax=ax, theme=theme, dpi=dpi, bulge=False, padding=0.3,
                         wire_sep=0.5, wire_label=wire_labels)
        _label_meters(diagram, ax)

        # Reuse the theme's wire color for the titles and separators.
        color = ax.lines[-1].get_color()

        # QuTiP's wire runs from zero to the right edge minus its padding.
        # Trim the outer margins where the panels meet.
        left, right = ax.get_xlim()
        left = left if i == 0 else 0
        right -= 0.3

        # Make the panel wide enough for its title (about 0.085 units per character).
        width = max(right - left, len(name) * 0.085)
        if left + width > right:
            # Extend the existing quantum lines and both strokes of a bit wire.
            for line in ax.lines:
                xs, ys = line.get_data()
                if len(xs) == 2 and ys[0] == ys[1] and xs[0] == 0 and isclose(xs[-1], right):
                    line.set_xdata([0, left + width])

        ax.set_xlim(left, left + width)
        ax.set_title(name, color=color, fontsize=10, pad=5)

        # A dashed separator at the left edge of every panel after the first.
        if i:
            ax.axvline(0, color=color, linestyle="--", linewidth=0.9,
                       alpha=0.65, clip_on=False)
        widths.append(width)

    # Size each panel in proportion to its width, and the figure to fit them all.
    axes[0, 0].get_gridspec().set_width_ratios(widths)
    bottom, top = axes[0, 0].get_ylim()
    height = top - bottom
    fig.set_size_inches(sum(widths) + 0.4, height + 1.05)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.78, wspace=0)

    if title:
        fig.suptitle(title, color=color, fontsize=12)
    return fig


def draw_register(register, theme="qutip", dpi=150, save_path=None, show=True, title=None):
    # QuTiP draw() calls show() internally; retain the wrapper's show=False option.
    with patch("matplotlib.pyplot.show"):
        if any(event.kind == "region" for event in register.history):
            # Protocol rounds that marked regions are drawn as one panel per region.
            figure = _draw_regions(register, theme, dpi, title)
        else:
            diagram, labels = history_circuit(register, bundle_classical=True)
            diagram.draw("matplotlib", theme=theme, dpi=dpi, wire_label=labels,
                         title=title, bulge=False)
            figure = plt.gcf()
            _label_meters(diagram, figure.axes[0])
            figure.set_size_inches(figure.get_size_inches() + (0.5, 0.75))
            figure.tight_layout()

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

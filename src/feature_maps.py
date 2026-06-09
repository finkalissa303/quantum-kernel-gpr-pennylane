# -*- coding: utf-8 -*-
"""Pluggable quantum feature maps for the fidelity kernel.

A feature map defines the unitary U(x; theta) that prepares the data-encoding
state |phi(x; theta)> = U(x; theta)|0>. `QuantumKernel` composes a feature map
with the fidelity-kernel machinery, so trying a different circuit means writing
a new FeatureMap -- the kernel itself never changes.
"""

import torch
import pennylane as qml


class FeatureMap:
    """Interface for a parameterized quantum feature map.

    Subclasses set `n_qubits` and `n_params`, and implement `apply(x, params)`
    to enqueue the gates of U(x; theta) on the active PennyLane device."""

    n_qubits: int
    n_params: int

    def apply(self, x, params):
        raise NotImplementedError


class ChebyshevFeatureMap(FeatureMap):
    """Hardware-efficient, Chebyshev-inspired feature map (Rapp & Roth, Fig. 2).

    Initial RY rotations, then `n_layers` blocks of data-encoding RX rotations
    (whose angles reuse the RY parameters, per the paper) followed by a ring of
    CRZ entanglers.

    Parameters
    ----------
    n_qubits, n_layers : int
        Circuit width and number of repeated entangling layers.
    phi : callable
        Data-encoding function. Default torch.arccos (Chebyshev); its domain is
        [-1, 1], so inputs must be scaled into that range.
    scale : float
        Multiplier on the data-dependent rotation angle (the paper uses 2.0).
    """

    def __init__(self, n_qubits, n_layers, phi=torch.arccos, scale=2.0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.phi = phi
        self.scale = scale
        # n_qubits RY angles + n_qubits CRZ angles per layer; the RX gates
        # reuse the RY angles, so they require no extra parameters.
        self.n_params = n_qubits * (n_layers + 1)

    def apply(self, x, params):
        pairs = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        for j in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RX(params[i] * self.scale * self.phi(x), wires=i)
            for k, (q1, q2) in enumerate(pairs):
                qml.CRZ(params[self.n_qubits + j * len(pairs) + k], wires=[q1, q2])

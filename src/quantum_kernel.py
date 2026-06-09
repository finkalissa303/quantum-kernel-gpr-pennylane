# -*- coding: utf-8 -*-
"""Trainable quantum fidelity kernel for Gaussian Process Regression.

Implements the hardware-efficient, Chebyshev-inspired feature map and the
fidelity kernel  k(x, x') = |<phi(x')|phi(x)>|^2  from Rapp & Roth,
"Quantum Gaussian Process Regression for Bayesian Optimization"
(arXiv:2304.12923), Eqs. (6)-(8).
"""

import torch
import gpytorch
import pennylane as qml


class QuantumKernel(gpytorch.kernels.Kernel):
    """GPyTorch kernel whose entries are quantum state fidelities.

    The feature map U(x; theta) prepares |phi(x; theta)> = U(x; theta)|0>, and
    a kernel entry is the ground-state probability of U(x1) U^dagger(x2), i.e.
    the fidelity |<phi(x2)|phi(x1)>|^2 (paper Eq. 8).

    Parameters
    ----------
    n_qubits, n_layers : int
        Circuit width and number of repeated entangling layers.
    phi : callable
        Data-encoding function applied to each input before the RX rotation.
        Typically torch.arccos (Chebyshev feature map); note its domain is
        [-1, 1], so inputs must be scaled into that range.
    eig_cutoff : bool
        If True, project each symmetric Gram matrix onto the PSD cone by
        truncating negative eigenvalues at zero. This is the regularization
        described in the paper (Sec. 1.1) and matters most for (near-)noiseless
        targets; with appreciable likelihood noise it is usually unnecessary.
    """

    def __init__(self, n_qubits, n_layers, phi, eig_cutoff=False, **kwargs):
        super().__init__(**kwargs)

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.phi = phi
        self.eig_cutoff = eig_cutoff

        # n_qubits RY angles + n_qubits CRZ angles per layer. The RX
        # data-encoding gates deliberately reuse the RY angles (paper Fig. 2),
        # so they require no extra parameters.
        self.n_params = self.n_qubits * (self.n_layers + 1)
        self.params = torch.nn.Parameter(2 * torch.pi * torch.rand(self.n_params))

        # "default.qubit" is PennyLane's CPU statevector simulator. Swap this
        # for a hardware/plugin device to run on a real backend.
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.quantum_kernel_circuit = qml.QNode(
            self._circuit_definition, self.dev, interface="torch"
        )

    def feature_map(self, x, params):
        """Apply U(x; theta): initial RY rotations, then `n_layers` blocks of
        data-encoding RX rotations followed by a ring of CRZ entanglers."""
        pairs = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        for j in range(self.n_layers):
            for i in range(self.n_qubits):
                # The factor 2.0 widens the data-dependent rotation range.
                qml.RX(params[i] * 2.0 * self.phi(x), wires=i)
            for k, (q1, q2) in enumerate(pairs):
                qml.CRZ(params[self.n_qubits + j * len(pairs) + k], wires=[q1, q2])

    def _circuit_definition(self, x1, x2, params):
        """U(x1) then U^dagger(x2); return computational-basis probabilities."""
        self.feature_map(x1, params)
        qml.adjoint(self.feature_map)(x2, params)
        return qml.probs(wires=range(self.n_qubits))

    def kernel(self, x1, x2, params):
        """Fidelity = probability of the all-zeros state (paper Eq. 8).

        flatten() makes the extraction robust to whether the circuit was called
        with scalar inputs (probs shape (2**n,)) or with the trailing (n, 1)
        feature dimension GPyTorch supplies (probs shape (1, 2**n))."""
        probs = self.quantum_kernel_circuit(x1, x2, params)
        return probs.flatten()[0]

    def _project_psd(self, K):
        """Truncate the Gram-matrix spectrum at zero (projection onto PSD)."""
        evals, evecs = torch.linalg.eigh(K)
        evals = torch.clamp(evals, min=0.0)
        return (evecs * evals) @ evecs.transpose(-1, -2)

    def forward(self, x1, x2, diag=False, **params):
        """Assemble the kernel matrix K[i, j] = kernel(x1[i], x2[j]).

        The symmetric shortcut (compute the upper triangle, mirror it, and use
        the unit diagonal) is only valid when x1 and x2 are the *same* points,
        so it is gated on torch.equal rather than on equal length -- otherwise
        a cross-covariance K(A, B) with len(A) == len(B) gets corrupted."""
        if diag:
            # k(x, x) = |<phi(x)|phi(x)>|^2 = 1 exactly, since U(x) U^dagger(x) = I.
            return torch.ones(len(x1), dtype=x1.dtype, device=x1.device)

        symmetric = x1.shape == x2.shape and torch.equal(x1, x2)
        kernel_matrix = torch.zeros(
            (len(x1), len(x2)), dtype=x1.dtype, device=x1.device
        )

        for i in range(len(x1)):
            if symmetric:
                kernel_matrix[i, i] = 1.0
                start_j = i + 1
            else:
                start_j = 0

            for j in range(start_j, len(x2)):
                val = self.kernel(x1[i], x2[j], self.params)
                kernel_matrix[i, j] = val
                if symmetric:
                    kernel_matrix[j, i] = val

        if symmetric and self.eig_cutoff:
            kernel_matrix = self._project_psd(kernel_matrix)

        return kernel_matrix

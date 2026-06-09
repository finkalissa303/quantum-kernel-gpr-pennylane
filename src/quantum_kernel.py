# -*- coding: utf-8 -*-
"""Trainable quantum fidelity kernel for Gaussian Process Regression.

Implements the fidelity kernel  k(x, x') = |<phi(x')|phi(x)>|^2  (Rapp & Roth,
"Quantum Gaussian Process Regression for Bayesian Optimization",
arXiv:2304.12923, Eqs. 6-8) on top of a pluggable feature map. The feature map
defines the circuit; this class only provides the kernel machinery.
"""

import torch
import gpytorch
import pennylane as qml


class QuantumKernel(gpytorch.kernels.Kernel):
    """GPyTorch kernel whose entries are quantum state fidelities.

    A kernel entry is the ground-state probability of U(x1) U^dagger(x2), i.e.
    the fidelity |<phi(x2)|phi(x1)>|^2 (paper Eq. 8), where U is supplied by
    `feature_map`.

    Parameters
    ----------
    feature_map : FeatureMap
        Defines U(x; theta); owns `n_qubits` and `n_params`.
    device : str
        PennyLane device name (default "default.qubit", the CPU statevector
        simulator). Pass another backend to run elsewhere.
    eig_cutoff : bool
        If True, project each symmetric Gram matrix onto the PSD cone by
        truncating negative eigenvalues at zero. This is the regularization
        described in the paper (Sec. 1.1) and matters most for (near-)noiseless
        targets; with appreciable likelihood noise it is usually unnecessary.
    """

    def __init__(self, feature_map, device="default.qubit", eig_cutoff=False, **kwargs):
        super().__init__(**kwargs)

        self.feature_map = feature_map
        self.eig_cutoff = eig_cutoff

        self.n_params = feature_map.n_params
        self.params = torch.nn.Parameter(2 * torch.pi * torch.rand(self.n_params))

        self.dev = qml.device(device, wires=feature_map.n_qubits)
        self.quantum_kernel_circuit = qml.QNode(
            self._circuit_definition, self.dev, interface="torch"
        )

    def _circuit_definition(self, x1, x2, params):
        """U(x1) then U^dagger(x2); return computational-basis probabilities."""
        self.feature_map.apply(x1, params)
        qml.adjoint(self.feature_map.apply)(x2, params)
        return qml.probs(wires=range(self.feature_map.n_qubits))

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

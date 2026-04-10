# Abstract

Topological deep learning is often presented as if its main value were expressive novelty. This benchmark tests a narrower and more defensible claim: topology may improve sample efficiency and robustness when the target function is aligned with a structured graph prior.

We provide three synthetic evaluation regimes: a spurious-topology setting, a homology-aligned setting, and an anti-topology setting. We compare a topology-aware message passing baseline against standard GCN, GAT, and Transformer baselines under clean and degree-preserving rewired conditions.

The benchmark reports clean accuracy, rewired accuracy, robustness gap, multi-seed mean and standard deviation, and spectral probes such as algebraic connectivity and Dirichlet energy. The goal is not to claim a new computational class, but to measure when topology acts as a useful inductive bias and when it does not.

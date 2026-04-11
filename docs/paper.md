# Paper Skeleton

## Title
Topological Inductive Bias in Graph Learning Under Controlled Distribution Shift

## Abstract
This benchmark tests whether topological inductive bias improves sample efficiency and robustness under degree-preserving rewiring. We evaluate standard graph baselines and a topology-aware message passing model across synthetic tasks designed to isolate structural signal from spurious correlation.

## 1. Introduction
- Why inductive bias matters
- Why topology is often overstated
- Why controlled synthetic tasks are useful

## 2. Problem Setup
- Tasks
- Models
- Metrics
- Distribution shift protocol

## 3. Benchmark Design
- Spurious topology
- Homology-aligned task
- Anti-topology task
- Rewiring stress test

## 4. Experimental Protocol
- Multi-seed runs
- Sample efficiency sweep
- Rewiring sweep
- Statistical reporting

## 5. Results
- Clean accuracy
- Rewired accuracy
- Robustness gap
- Learning curves

## 6. Discussion
- When topology helps
- When it fails
- Limitations
- Future work

## 7. Reproducibility
- Deterministic seeds
- Minimal dependencies
- Saved JSON outputs

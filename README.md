# Adaptive Hybrid Thermal Digital Twin Framework for PMSM

## Project Overview

This project develops an **adaptive hybrid thermal digital twin framework** for **Permanent Magnet Synchronous Motors (PMSMs)**.

The framework combines:

- physics-based thermal modeling
- numerical ODE simulation
- parameter calibration
- residual machine learning correction
- adaptive confidence gating

The goal is to create a robust, interpretable, and deployable framework for:

1. **Thermal Virtual Sensing**
2. **Predictive Maintenance**

This work builds upon existing PMSM thermal digital twin literature but introduces a different architectural philosophy.

---

# Motivation

Existing PMSM thermal prediction approaches typically fall into three categories:

### Pure Physics Models

Examples:

- Lumped Parameter Thermal Networks (LPTN)
- Reduced-order thermal models
- Kalman filtering
- Thermal observers

Advantages:

- physically interpretable
- lightweight
- deployable

Limitations:

- parameter uncertainty
- missing physics
- simplified heat transfer assumptions
- reduced performance under complex dynamics

---

### Pure Machine Learning Models

Examples:

- MLP
- GRU
- LSTM
- Deep black-box estimators

Advantages:

- nonlinear modeling capability

Limitations:

- weak interpretability
- poor extrapolation
- limited physical consistency
- deployment concerns

---

### Existing Hybrid Models

Recent papers combine physics and ML.

Examples:

- DBBC + HPA-Net frameworks
- Bayesian calibrated thermal twins
- physics-aware neural networks

However, many existing methods use:

- direct neural prediction
- black-box refinement
- limited numerical diagnostics

---

# Our Research Philosophy

We do **NOT** replace physics with a neural network.

We do **NOT** stop at parameter calibration.

We use:

## Physics Backbone → Calibration → Diagnostics → Adaptive Residual Correction

Core principle:

Physics should provide the primary prediction.

Machine learning should only learn **structured model mismatch**.

---

# Framework Architecture

## Stage 1 — Data Layer

Input variables:

### Electrical signals

- id
- iq
- ud
- uq

### Operating variables

- rotor speed
- torque
- ambient temperature

### Thermal measurements

- stator winding temperature
- stator temperature
- PM / rotor temperature

### Derived features

- copper losses
- iron losses
- PM losses
- power features

Output:

supervised thermal prediction dataset.

---

## Stage 2 — Physics Thermal Backbone

We use a **Reduced Order Thermal Model / LPTN (Lumped Parameter Thermal Network)**.

Thermal states:

```math
T = [T_winding, T_stator, T_PM]
```

General form:

```math
C \dot{T} = f(T,Q,R)
```

where:

* thermal resistances model heat transfer
* thermal capacitances model thermal storage
* heat sources model generated losses

Outputs:

```math
T_{phys}
```

This is the thermal equivalent of the dq electrical equations used in our earlier electrical digital twin framework.

---

## Stage 3 — Numerical Solver Layer

We explicitly solve the thermal ODEs.

Primary solver:

* RK4 (Runge-Kutta 4th Order)

Future comparisons:

* Radau
* BDF

Purpose:

* stable numerical integration
* transparent solver behavior
* reproducible thermal trajectories

Output:

```math
T_{phys}(t)
```

---

## Stage 4 — Parameter Calibration Layer

We use:

## Particle Swarm Optimization (PSO)

instead of Bayesian DBBC calibration.

Unknown parameters:

* thermal resistances
* thermal capacitances
* heat transfer coefficients

Optimization objective:

```math
MSE(T_{measured}, T_{phys})
```

Goal:

reduce baseline physics mismatch.

Output:

calibrated physics model.

---

## Stage 5 — Numerical Diagnostic Framework

This is a major component of our methodology.

### D1 — Timestep Sensitivity

Evaluate:

* timestep dependence
* integration stability

---

### D2 — Solver Comparison

Compare:

* RK4
* Radau
* BDF

Purpose:

identify numerical stiffness and solver dependence.

---

### D3 — Residual Structure Analysis

Compute:

```math
Residual = T_{measured} - T_{phys}
```

Analyze:

* systematic bias
* drift
* operating regime failures
* missing thermal physics

---

### D4 — Dynamic Regime Segmentation

Compare performance across:

* steady state
* transient load changes
* thermal cooldown
* aggressive operating cycles

---

## Stage 6 — Residual Learning Layer

Unlike direct black-box prediction methods, we explicitly learn:

```math
\Delta T = T_{measured} - T_{phys}
```

Machine learning learns **only the missing physics**.

Inputs:

### Raw features

* currents
* voltages
* speed
* torque
* ambient temperature

### Physics features

* calibrated thermal predictions
* temperature derivatives

### Dynamic indicators

* power derivatives
* speed derivatives
* thermal dynamic score

### Confidence features

* rolling residual magnitude

Initial model:

### Residual MLP

Future candidates:

* LSTM
* Temporal CNN
* Mixture of Experts

Output:

```math
\Delta T_{ML}
```

---

## Stage 7 — Alpha-Gated Adaptive Trust Layer

This is the signature component of the framework.

We learn an adaptive trust coefficient:

```math
\alpha \in [0,1]
```

Behavior:

Physics performing well:

```math
\alpha \downarrow
```

Physics struggling:

```math
\alpha \uparrow
```

Final hybrid prediction:

```math
T_{hybrid}
=
T_{phys}
+
\alpha \Delta T_{ML}
```

This creates adaptive blending between:

* physics prediction
* learned correction

---

## Stage 8 — Evaluation and Benchmarking

Benchmarks:

### Baseline 1

Raw thermal physics model.

### Baseline 2

PSO calibrated thermal model.

### Baseline 3

Existing literature framework.

(DBBC + HPA-Net)

### Baseline 4

Proposed adaptive hybrid framework.

Metrics:

* RMSE
* MAE
* transient accuracy
* regime robustness
* generalization performance
* computational efficiency

---

# Applications

## Thermal Virtual Sensing

Estimate hidden thermal states:

* PM temperature
* rotor thermal state
* internal temperatures

without direct sensing.

---

## Predictive Maintenance

Use thermal predictions for:

* overheating detection
* cooling degradation monitoring
* demagnetization risk
* anomaly detection
* health monitoring

---

# Codebase Philosophy

Modular architecture.

Core pipeline:

```plaintext
Data
	↓
Thermal Physics Model
	↓
RK4 Solver
	↓
PSO Calibration
	↓
Diagnostics
	↓
Residual Learning
	↓
Alpha Gate
	↓
Hybrid Thermal Prediction
	↓
Virtual Sensing / Predictive Maintenance
```

---

# Implementation Roadmap

Phase 1:

* dataset setup
* thermal physics model
* RK4 solver

Phase 2:

* PSO calibration
* baseline validation

Phase 3:

* D1–D4 diagnostics

Phase 4:

* residual learning implementation

Phase 5:

* alpha gate integration

Phase 6:

* benchmarking vs literature

---

## Development Goal

Build a **modular, interpretable, and deployable adaptive thermal digital twin framework** for PMSM systems capable of operating across dynamic conditions and supporting industrial virtual sensing and predictive maintenance applications.

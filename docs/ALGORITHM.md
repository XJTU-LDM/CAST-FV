# CAST-FV algorithm notes

This document defines the public algorithmic boundary implemented in `src/castfv`. It is deliberately independent of CFD solution labels and reference trajectories.

## 1. Neural state representation

On a Cartesian cell-centered mesh, a neural map $f_{\theta}$ receives normalized coordinates and boundary-condition channels and returns the complete physical state:

$$
\mathbf{q}_{\theta}=f_{\theta}(\mathbf{x},\mathbf{b}).
$$

The public state vectors are

$$
\mathbf{q}^{2D}=(p,u,v,T),
\qquad
\mathbf{q}^{3D}=(p,u,v,w,T).
$$

The network therefore parameterizes the unknown discrete state; it is not trained to imitate a stored CFD solution.

## 2. Boundary-face construction

Dirichlet values are imposed algebraically on boundary faces. If $\phi_P$ is a cell-center value and $\phi_b$ is prescribed at the adjacent boundary face, the ghost value is

$$
\phi_G=2\phi_b-\phi_P.
$$

This relation lets the same centered face and diffusion formulas be used at interior and boundary locations. Normal pressure gradients are set through a zero-gradient pressure construction, while a mean-pressure penalty removes the arbitrary pressure gauge.

## 3. Finite-volume residual

For a control volume $P$, the continuity residual is

$$
R_c=\sum_{f\in\partial P}(\mathbf{u}_f\cdot\mathbf{n}_f)A_f.
$$

The momentum residual in direction $i$ is

$$
R_{m_i}=R_{t,i}
+\sum_{f\in\partial P}(\mathbf{u}_f\cdot\mathbf{n}_f)u_{i,f}A_f
+\sum_{f\in\partial P}p_f n_{i,f}A_f
-\frac{1}{Re}\sum_{f\in\partial P}(\nabla u_i)_f\cdot\mathbf{n}_f A_f.
$$

The passive-scalar residual is

$$
R_T=R_{t,T}
+\sum_{f\in\partial P}(\mathbf{u}_f\cdot\mathbf{n}_f)T_fA_f
-\frac{1}{Pe}\sum_{f\in\partial P}(\nabla T)_f\cdot\mathbf{n}_f A_f.
$$

For steady construction, $R_t=0$. For physical-time advancement from the retained state $\mathbf{q}^{n-1}$, backward Euler supplies

$$
R_t=\frac{\mathbf{q}^{n}_{\theta}-\mathbf{q}^{n-1}}{\Delta t}V_P
$$

for the transported velocity and scalar components.

## 4. Optimization objective

The common numerical objective is the unweighted sum of component-wise mean absolute residuals plus a pressure-gauge penalty:

$$
\mathcal{J}(\theta)
=\langle |R_c|\rangle
+\sum_i\langle |R_{m_i}|\rangle
+\langle |R_T|\rangle
+\lambda_p\langle p\rangle^2.
$$

AdamW updates $\theta$ for a fixed budget. Every finite iterate is eligible for retention, and the state with the smallest encountered $\mathcal{J}$ is returned.

## 5. Physical-time advancement

At every requested physical-time level $n$:

```text
create a fresh neural map f_theta and fresh AdamW optimizer
repeat for the fixed update budget:
    q_candidate = f_theta(coordinates, boundary channels)
    build boundary faces algebraically
    assemble FV residuals using retained q_(n-1)
    update theta with AdamW
    retain q_candidate if its finite objective is the best so far
set q_(n) to the retained candidate
```

Only $\mathbf{q}^{n-1}$ crosses the physical-time boundary. Neural weights, optimizer moments, and rejected candidates do not.

## 6. Implementation map

| Numerical role | Public implementation |
|---|---|
| Case and optimization contracts | `src/castfv/config.py` |
| Coordinates and boundary channels | `src/castfv/features.py` |
| Neural state parameterizations | `src/castfv/models.py` |
| Boundary algebra and FV residuals | `src/castfv/residuals.py` |
| Fixed-budget optimization and retention | `src/castfv/optimize.py` |
| Reproducible artifacts | `src/castfv/output.py` |
| Command-line entry points | `src/castfv/cli.py` |

## 7. Public exclusions

This implementation does not contain pressure projection, persistent face-flux states, accepted-state logic, pretrained neural operators, CFD reference readers, paper result datasets, or other solver extensions outside the present method.

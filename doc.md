
# Implementation notes for the pseudo-spectral incompressible Navier–Stokes solver

## 1. Scope

This document summarizes the pseudo-spectral Navier–Stokes method described in Section 2.2 of the thesis, in a form intended to guide a later implementation.

The method solves the incompressible Navier–Stokes equations in a triply periodic box:

```math
\frac{\partial u_i}{\partial t}
+
\frac{\partial}{\partial x_j}(u_j u_i)
=
-\frac{\partial P}{\partial x_i}
+
\nu \nabla^2 u_i
```

with

```math
\nabla \cdot u = 0.
```

The pressure is not solved explicitly. Instead, after each time-integration substep, the velocity field is projected onto the divergence-free subspace in Fourier space.

The method assumes:

* a Cartesian periodic domain;
* uniform grid spacing;
* Fourier representation in all three spatial directions;
* velocity components stored either in physical space `u(x)` or spectral space `u_hat(k)`;
* nonlinear products evaluated in physical space;
* spatial derivatives evaluated exactly in spectral space;
* explicit third-order low-storage Runge–Kutta time stepping;
* dealiasing by phase shifting plus ellipsoidal truncation.

---

## 2. Fourier representation

The velocity field is expanded as

```math
u(x,t) = \sum_k \hat{u}(k,t) \exp(i k \cdot x).
```

Here:

```math
x = (x,y,z),
```

and

```math
k = (k_x,k_y,k_z).
```

A spatial derivative is computed spectrally using

```math
\frac{\partial^n g}{\partial x^n}
\longleftrightarrow
(i k)^n \hat{g}(k).
```

Therefore:

```math
\nabla g
\longleftrightarrow
i k \hat{g},
```

and

```math
\nabla^2 g
\longleftrightarrow
-|k|^2 \hat{g}.
```

For velocity component `u_i`, the semi-discrete spectral equation can be written as

```math
\left(
\frac{d}{dt}
+
\nu |k|^2
\right)
\hat{u}_i(k)
=
\hat{f}_i(k),
```

where the nonlinear term is

```math
\hat{f}_i(k)
=
-\widehat{
\frac{\partial}{\partial x_j}(u_j u_i)
}.
```

Equivalently,

```math
\hat{f}_i(k)
=
- i k_j \widehat{u_j u_i}(k).
```

The summation convention over repeated indices is used.

---

## 3. Data layout

A practical implementation should maintain:

```text
u_hat[3, Nx, Ny, Nz_complex]
u_phys[3, Nx, Ny, Nz]
rhs_hat[3, Nx, Ny, Nz_complex]
```

If using real-to-complex FFTs, the last spectral dimension has size `Nz/2 + 1`.

For a fully complex implementation, all dimensions have size `Nx`, `Ny`, `Nz`.

The three velocity components are stored collocated in the physical grid. This differs from the finite-difference MAC solver described earlier in the thesis.

---

## 4. Wavenumber arrays

For a domain of size

```math
L_x \times L_y \times L_z,
```

define

```math
k_x = \frac{2\pi}{L_x} n_x,
```

with integer Fourier mode index `n_x`.

For an FFT ordering compatible with most libraries:

```python
kx = 2*pi * fftfreq(Nx, d=Lx/Nx)
ky = 2*pi * fftfreq(Ny, d=Ly/Ny)
kz = 2*pi * fftfreq(Nz, d=Lz/Nz)
```

For a real-to-complex FFT in the `z` direction:

```python
kz = 2*pi * rfftfreq(Nz, d=Lz/Nz)
```

Construct broadcastable arrays:

```text
KX, KY, KZ
K2 = KX**2 + KY**2 + KZ**2
```

Special treatment is required for the zero mode:

```text
K2[0,0,0] = 0
```

When dividing by `K2`, avoid division at `k = 0`.

---

## 5. Spectral projection to enforce incompressibility

At the end of every Runge–Kutta substep, project the velocity field onto the divergence-free subspace.

Given an arbitrary spectral velocity field

```math
\hat{u}(k),
```

define the solenoidal field

```math
\hat{u}^{s}(k)
=
\hat{u}(k)
-
k
\frac{\hat{u}(k)\cdot k}{|k|^2}.
```

Componentwise:

```math
\hat{u}^{s}_i(k)
=
\hat{u}_i(k)
-
k_i
\frac{k_j \hat{u}_j(k)}{|k|^2}.
```

For `k = 0`, leave the mode unchanged or set it according to the desired mean velocity. For decaying turbulence with zero mean, it is common to set the zero mode to zero.

Implementation sketch:

```python
def project_div_free(u_hat, KX, KY, KZ, K2):
    dot = KX*u_hat[0] + KY*u_hat[1] + KZ*u_hat[2]

    factor = zeros_like(dot)
    mask = K2 != 0
    factor[mask] = dot[mask] / K2[mask]

    u_hat[0] -= KX * factor
    u_hat[1] -= KY * factor
    u_hat[2] -= KZ * factor

    return u_hat
```

This projection accounts for the pressure gradient and avoids solving a Poisson equation for pressure.

---

## 6. Nonlinear term

The nonlinear term is written in conservative form:

```math
f_i
=
-\frac{\partial}{\partial x_j}(u_j u_i).
```

In spectral space:

```math
\hat{f}_i(k)
=
- i k_j \widehat{u_j u_i}(k).
```

A direct convolution in Fourier space is too expensive. Instead:

1. transform `u_hat` to physical space;
2. compute products `u_j * u_i` in physical space;
3. transform those products back to spectral space;
4. apply the derivative spectrally by multiplying by `i k_j`;
5. sum over `j`.

For each component `i`:

```math
\hat{f}_i
=
-
i k_x \widehat{u_x u_i}
-
i k_y \widehat{u_y u_i}
-
i k_z \widehat{u_z u_i}.
```

Implementation sketch without dealiasing:

```python
def nonlinear_rhs_conservative(u_hat):
    u = ifft(u_hat)  # shape: (3, Nx, Ny, Nz)

    rhs_hat = zeros_like(u_hat)

    for i in range(3):
        ux_ui_hat = fft(u[0] * u[i])
        uy_ui_hat = fft(u[1] * u[i])
        uz_ui_hat = fft(u[2] * u[i])

        rhs_hat[i] = (
            -1j*KX*ux_ui_hat
            -1j*KY*uy_ui_hat
            -1j*KZ*uz_ui_hat
        )

    return rhs_hat
```

In practice, this must be replaced by the dealiased version described below.

---

## 7. Viscous term

The viscous term is diagonal in Fourier space:

```math
\nu \nabla^2 u_i
\longleftrightarrow
-\nu |k|^2 \hat{u}_i.
```

Thus the full right-hand side can be assembled as

```math
H_i(\hat{u})
=
\hat{f}_i
-
\nu |k|^2 \hat{u}_i.
```

Depending on the final design, viscosity can be handled explicitly inside the RK3 scheme or through an integrating factor. Section 2.2 presents the equation in the form

```math
\left(\frac{d}{dt}+\nu |k|^2\right)\hat{u} = \hat{f},
```

but the RK3 algorithm is described generically for

```math
\frac{du}{dt} = H(u,t).
```

A direct implementation can therefore start with the fully explicit RHS:

```python
rhs_hat = nonlinear_hat - nu * K2 * u_hat
```

For high viscosity or very fine grids, an integrating-factor or semi-implicit treatment could be added later.

---

## 8. Low-storage Williamson RK3 time stepping

The thesis uses the classical low-storage third-order Runge–Kutta variant of Williamson.

For

```math
\frac{du}{dt}=H(u,t),
```

the algorithm is:

```text
U = u^n

G = H(U, t^n)
U = U + (1/3) dt G
project(U)

G = -(5/9) G + H(U, t^n + dt/3)
U = U + (15/16) dt G
project(U)

G = -(153/128) G + H(U, t^n + 3dt/4)
u^{n+1} = U + (8/15) dt G
project(u^{n+1})
```

The projection should be applied after each substep.

Implementation sketch:

```python
def step_rk3(u_hat, t, dt):
    U = u_hat.copy()

    G = H(U, t)
    U = U + (1.0/3.0) * dt * G
    U = project_div_free(U)

    G = -(5.0/9.0) * G + H(U, t + dt/3.0)
    U = U + (15.0/16.0) * dt * G
    U = project_div_free(U)

    G = -(153.0/128.0) * G + H(U, t + 3.0*dt/4.0)
    U = U + (8.0/15.0) * dt * G
    U = project_div_free(U)

    return U
```

---

## 9. Aliasing errors

Because nonlinear products are evaluated in physical space, pseudo-spectral methods suffer from aliasing errors.

In one dimension, the Fourier transform of a product computed on a finite grid contains the desired alias-free convolution plus an aliasing contribution:

```math
c^{aliased}_k
=
\sum_{n+m=k} \hat{a}_n \hat{b}_m
+
\sum_{n+m=k \pm M} \hat{a}_n \hat{b}_m.
```

The first sum is the true convolution contribution. The second sum is the aliasing error.

Aliasing errors are dangerous because they can produce spurious energy transfer, including artificial energy injection. In the inviscid Euler case, this can lead to numerical blow-up.

---

## 10. Phase-shift dealiasing principle

If a field is shifted in physical space by `Delta`, each Fourier mode is multiplied by

```math
\exp(i n \Delta).
```

For a product evaluated on a shifted grid and shifted back to the original grid, the alias-free contribution is unchanged, while the aliasing contribution is multiplied by

```math
\exp(\pm i M \Delta).
```

Choosing

```math
\Delta = \frac{\pi}{M}
```

corresponds to a half-grid shift and gives

```math
\exp(\pm i M \Delta) = -1.
```

Then the dealiased product is obtained by averaging the product computed on the original grid and the product computed on the shifted grid:

```math
c_k
=
\frac{1}{2}
\left(
c^{aliased}_k
+
c^{shifted}_k
\right).
```

In one dimension, this exactly cancels the aliasing error.

---

## 11. Practical 3D dealiasing used in the thesis

A full 3D phase-shift dealiasing would require evaluating each product on eight shifted grids. This is expensive.

The method described in Section 2.2 instead uses:

1. two evaluations of each nonlinear product:

   * one on the original grid;
   * one on a shifted grid;
2. an ellipsoidal truncation of high-order Fourier modes.

This is a partial dealiasing strategy. It does not remove every aliasing contribution exactly in 3D, but strongly reduces aliasing errors and was found sufficient for energy-conserving DNS/LES in the thesis.

---

## 12. Ellipsoidal truncation

After nonlinear terms are computed, high modes are truncated according to the ellipsoidal criterion

```math
\frac{k_x^2}{(n_x/2)^2}
+
\frac{k_y^2}{(n_y/2)^2}
+
\frac{k_z^2}{(n_z/2)^2}
\leq 1.
```

Modes outside this ellipsoid are zeroed.

In implementation, this criterion should be applied in terms of integer Fourier mode indices rather than dimensional wavenumbers.

Let

```text
mx, my, mz
```

be the integer Fourier indices associated with each spectral coefficient. Then define

```python
ellipsoid_mask = (
    (mx**2)/(Nx/2)**2
    + (my**2)/(Ny/2)**2
    + (mz**2)/(Nz/2)**2
    <= 1.0
)
```

After computing the nonlinear RHS:

```python
rhs_hat[:, ~ellipsoid_mask] = 0
```

It is also reasonable to apply the same truncation to `u_hat` after projection, especially after operations that may populate high modes.

---

## 13. Shifted-grid product implementation

For two fields `a_hat` and `b_hat`, a dealiased product can be computed as follows.

### 13.1 Original product

```python
a = ifft(a_hat)
b = ifft(b_hat)
c0_hat = fft(a * b)
```

### 13.2 Shifted product

Choose a half-cell shift. For a shift vector

```math
\Delta = (\Delta_x,\Delta_y,\Delta_z),
```

the shifted spectral fields are

```math
\hat{a}^{shift}(k)
=
\hat{a}(k) \exp(i k \cdot \Delta),
```

```math
\hat{b}^{shift}(k)
=
\hat{b}(k) \exp(i k \cdot \Delta).
```

Then:

```python
phase = exp(1j * (KX*dx_shift + KY*dy_shift + KZ*dz_shift))

a_shift = ifft(a_hat * phase)
b_shift = ifft(b_hat * phase)

c_shift_hat = fft(a_shift * b_shift)
```

The shifted product must be shifted back:

```python
c_shift_back_hat = c_shift_hat * conj(phase)
```

Then average:

```python
c_hat = 0.5 * (c0_hat + c_shift_back_hat)
```

Finally apply the ellipsoidal truncation:

```python
c_hat[~ellipsoid_mask] = 0
```

A natural starting choice is a half-grid shift in all three directions:

```python
dx_shift = 0.5 * Lx / Nx
dy_shift = 0.5 * Ly / Ny
dz_shift = 0.5 * Lz / Nz
```

The exact convention must be kept consistent with the FFT normalization and the sign convention used by the FFT library.

---

## 14. Dealiased nonlinear RHS

For each product `u_j u_i`, compute the dealiased product:

```python
def product_dealiased(a_hat, b_hat):
    a = ifft(a_hat)
    b = ifft(b_hat)
    c0_hat = fft(a * b)

    phase = exp(1j * (KX*dx_shift + KY*dy_shift + KZ*dz_shift))

    a_s = ifft(a_hat * phase)
    b_s = ifft(b_hat * phase)
    cs_hat = fft(a_s * b_s)

    cs_hat = cs_hat * conj(phase)

    c_hat = 0.5 * (c0_hat + cs_hat)

    c_hat[~ellipsoid_mask] = 0

    return c_hat
```

Then assemble:

```python
def nonlinear_rhs_dealiased(u_hat):
    rhs = zeros_like(u_hat)

    for i in range(3):
        u0_ui_hat = product_dealiased(u_hat[0], u_hat[i])
        u1_ui_hat = product_dealiased(u_hat[1], u_hat[i])
        u2_ui_hat = product_dealiased(u_hat[2], u_hat[i])

        rhs[i] = (
            -1j*KX*u0_ui_hat
            -1j*KY*u1_ui_hat
            -1j*KZ*u2_ui_hat
        )

    rhs[:, ~ellipsoid_mask] = 0

    return rhs
```

Then the full RHS is:

```python
def H(u_hat, t):
    rhs = nonlinear_rhs_dealiased(u_hat)
    rhs -= nu * K2 * u_hat

    rhs = project_div_free(rhs)

    return rhs
```

Projection of the RHS is not strictly a replacement for projecting the velocity after each RK substep, but it is usually useful to keep the RHS in the solenoidal subspace. The main required operation remains projection of the velocity field after every RK substep.

---

## 15. Energy diagnostics

The average kinetic energy is

```math
E
=
\left\langle
\frac{u_i u_i}{2}
\right\rangle.
```

In physical space:

```python
E = mean(0.5 * (u[0]**2 + u[1]**2 + u[2]**2))
```

For a correctly normalized FFT, the equivalent spectral expression can also be used through Parseval’s identity.

The dissipation rate is

```math
\epsilon
=
\nu \left\langle 2 S_{ij} S_{ij} \right\rangle.
```

For incompressible periodic flows, this can also be computed from enstrophy:

```math
\epsilon
=
\nu \langle \omega_i \omega_i \rangle.
```

The vorticity is computed spectrally:

```math
\hat{\omega}
=
i k \times \hat{u}.
```

Implementation:

```python
omega_hat[0] = 1j*(KY*u_hat[2] - KZ*u_hat[1])
omega_hat[1] = 1j*(KZ*u_hat[0] - KX*u_hat[2])
omega_hat[2] = 1j*(KX*u_hat[1] - KY*u_hat[0])
omega = ifft(omega_hat)

epsilon = nu * mean(omega[0]**2 + omega[1]**2 + omega[2]**2)
```

---

## 16. Resolution criterion

For DNS, the thesis uses the usual Kolmogorov resolution criterion:

```math
k_c \eta \gtrsim 1.5.
```

Here

```math
\eta =
\left(
\frac{\nu^3}{\epsilon}
\right)^{1/4}
```

is the Kolmogorov scale.

For a grid with spacing `h`, the cutoff wavenumber is approximately

```math
k_c = \frac{\pi}{h}.
```

For a periodic box with `N` grid points and length `L`:

```math
h = \frac{L}{N},
```

so

```math
k_c = \frac{\pi N}{L}.
```

For the Taylor–Green vortex at `Re = 1600`, the thesis reports:

```text
N = 96:   kc eta = 0.56
N = 192:  kc eta = 1.12
N = 256:  kc eta = 1.5
```

Thus, `256^3` is the properly resolved DNS reference grid for that case.

---

## 17. Taylor–Green validation case

A useful validation case is the Taylor–Green vortex in a periodic box.

Initial condition:

```math
u =
U
\sin\left(\frac{2\pi x}{L}\right)
\cos\left(\frac{2\pi y}{L}\right)
\cos\left(\frac{2\pi z}{L}\right),
```

```math
v =
-U
\cos\left(\frac{2\pi x}{L}\right)
\sin\left(\frac{2\pi y}{L}\right)
\cos\left(\frac{2\pi z}{L}\right),
```

```math
w = 0.
```

The Reynolds number is defined as

```math
Re =
\frac{U L}{2\pi \nu}.
```

For the validation described in the thesis:

```text
Re = 1600
N = 256^3 for fully resolved DNS
periodic box
```

Important diagnostics:

* kinetic energy evolution;
* dissipation evolution;
* energy spectra;
* comparison against reference DNS;
* resolution criterion `kc eta`.

The dissipation should agree well with the reference solution for `256^3`. Lower resolutions show visible discrepancies, especially around and after the dissipation peak.

---

## 18. Inviscid time-reversibility test

The thesis also discusses a time-reversibility benchmark for the incompressible Euler equations, obtained by setting

```math
\nu = 0.
```

The equations are then

```math
\frac{\partial u_i}{\partial t}
+
\frac{\partial}{\partial x_j}(u_j u_i)
=
-\frac{\partial P}{\partial x_i},
```

```math
\nabla \cdot u = 0.
```

If

```math
(u(x,t), P(x,t))
```

is a solution, then

```math
(u(x,-t), P(x,-t))
```

is also a solution.

A numerical scheme should approximately preserve this property if it is sufficiently accurate and energy conserving.

Test procedure:

1. initialize an inviscid Taylor–Green vortex;
2. integrate forward until `t = 10`;
3. reverse time integration;
4. integrate back to the initial time;
5. check whether the initial diagnostics are recovered.

Diagnostics used:

```math
E
```

kinetic energy,

```math
\mathcal{E}
```

enstrophy,

and the 3D skewness-like diagnostic

```math
S_{3D}
=
\left\langle
\frac{\partial u_i}{\partial x_k}
\frac{\partial u_j}{\partial x_k}
S_{ij}
\right\rangle.
```

The thesis reports that:

* without dealiasing, the simulation blows up due to spurious energy injection;
* random phase shift can also blow up in this reversibility test because aliasing errors are not reduced consistently in the forward and backward phases;
* the phase-shift technique plus ellipsoidal truncation gives good reversibility, with small remaining energy differences attributed mainly to time-stepping errors.

---

## 19. Homogeneous isotropic turbulence validation

The second validation target is decaying homogeneous isotropic turbulence at approximately

```math
Re_\lambda \approx 100.
```

The thesis uses:

```text
N = 256 Fourier modes in each direction
kc = 128
initial kc eta ≈ 1.1
```

As the turbulence decays, both energy and dissipation decrease, the Kolmogorov length scale increases, and the computation becomes better resolved.

Diagnostics:

* kinetic energy decay;
* dissipation;
* Taylor-scale Reynolds number;
* compensated energy spectra;
* `kc eta` evolution.

At later times, the thesis reports that `kc eta > 1.5`, and the compensated spectrum shows an inertial range roughly over

```text
4 <= k <= 20
```

for the studied case.

---

## 20. Minimal solver workflow

A minimal implementation should follow this structure:

```text
1. Define domain sizes Lx, Ly, Lz.
2. Define grid sizes Nx, Ny, Nz.
3. Build physical grid.
4. Build spectral wavenumber arrays KX, KY, KZ, K2.
5. Build integer Fourier-mode arrays for truncation.
6. Build ellipsoidal truncation mask.
7. Initialize velocity field in physical space.
8. Transform velocity to Fourier space.
9. Project initial velocity to divergence-free space.
10. Apply ellipsoidal truncation.
11. For each time step:
    a. advance one Williamson RK3 step;
    b. project after each RK substep;
    c. truncate high modes;
    d. compute diagnostics.
```

---

## 21. Minimal pseudocode

```python
initialize_grid()
initialize_wavenumbers()
initialize_ellipsoid_mask()

u = initial_condition()
u_hat = fft(u)

u_hat = project_div_free(u_hat)
u_hat[:, ~ellipsoid_mask] = 0

t = 0.0

while t < t_end:
    u_hat = rk3_step(u_hat, t, dt)

    u_hat = project_div_free(u_hat)
    u_hat[:, ~ellipsoid_mask] = 0

    if should_output:
        u = ifft(u_hat)
        E = compute_energy(u)
        eps = compute_dissipation(u_hat)
        kc_eta = compute_kc_eta(eps, nu)

        write_diagnostics(t, E, eps, kc_eta)

    t += dt
```

---

## 22. Important implementation checks

### Divergence check

After projection:

```math
k \cdot \hat{u}(k) \approx 0.
```

Numerically:

```python
div_hat = KX*u_hat[0] + KY*u_hat[1] + KZ*u_hat[2]
max_div = max(abs(div_hat))
```

This should remain close to roundoff.

### Energy behavior

For inviscid simulations with proper dealiasing, kinetic energy should remain nearly constant, with only time-integration errors.

For viscous simulations, energy should decay.

### Aliasing check

Run the inviscid Taylor–Green test:

* with dealiasing disabled: the simulation may become unstable;
* with phase-shift plus ellipsoidal truncation: energy behavior should improve significantly.

### Resolution check

Monitor

```math
k_c \eta.
```

For DNS-quality results, aim for

```math
k_c \eta \geq 1.5.
```

---

## 23. Main implementation decisions still open

The thesis gives the numerical strategy but does not prescribe all low-level implementation choices. The following decisions should be made explicitly during coding:

1. FFT normalization convention.
2. Real-to-complex versus complex-to-complex storage.
3. Whether viscosity is treated explicitly or with an integrating factor.
4. Exact shifted-grid convention in 3D.
5. Whether to project only the velocity after RK substeps or also the RHS.
6. Whether to use conservative, advective, or skew-symmetric nonlinear form.

The thesis implementation uses the conservative nonlinear form for the pseudo-spectral equation, but it also notes that skew-symmetric forms can reduce aliasing errors at similar cost.

---

## 24. Recommended first implementation path

For a first clean implementation:

1. Use a triply periodic cubic domain.
2. Use complex-to-complex FFTs first, even if less memory efficient.
3. Implement Taylor–Green initial condition.
4. Implement projection.
5. Implement explicit RK3.
6. Implement conservative nonlinear RHS without dealiasing.
7. Verify basic correctness on short, low-Re simulations.
8. Add ellipsoidal truncation.
9. Add shifted-grid product dealiasing.
10. Run inviscid Taylor–Green and check energy conservation.
11. Run viscous Taylor–Green at `Re = 1600`.
12. Compare dissipation evolution for `N = 96`, `192`, and `256`.

Once validated, optimize memory and switch to real-to-complex FFTs or MPI-parallel FFTs if needed.

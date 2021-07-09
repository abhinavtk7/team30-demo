# Copyright (C) 2021 Jørgen S. Dokken and Igor A. Baratta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import argparse
import os

import dolfinx
import dolfinx.io
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import sys
from typing import Callable, TextIO

from generate_team30_meshes import (domain_parameters, model_parameters,
                                    surface_map)
from utils import (DerivedQuantities2D, MagneticFieldProjection2D, XDMFWrapper,
                   update_current_density)


def solve_team30(single_phase: bool, num_phases: int, omega_u: np.float64, degree: np.int32,
                 form_compiler_parameters: dict = {}, jit_parameters: dict = {}, apply_torque: bool = False,
                 T_ext: Callable[[float], float] = lambda t: 0, outdir: str = "results", steps_per_phase: int = 100,
                 outfile: TextIO = sys.stdout, plot: bool = False, progress: bool = False, mesh_dir: str = "meshes",
                 xdmf_file: str = None):
    """
    Solve the TEAM 30 problem for a single or three phase engine.

    Parameters
    ==========
    single_phase
        If true run the single phase model, otherwise run the three phase model

    num_phases
        Number of phases to run the simulation for

    omega_u
        Angular speed of rotor (Used as initial speed if apply_torque is True)

    degree
        Degree of magnetic vector potential functions space

    form_compiler_parameters
        Parameters used in FFCx compilation of this form. Run `ffcx --help` at
        the commandline to see all available options. Takes priority over all
        other parameter values, except for `scalar_type` which is determined by
        DOLFINx.

    jit_parameters
        Parameters used in CFFI JIT compilation of C code generated by FFCx.
        See `python/dolfinx/jit.py` for all available parameters.
        Takes priority over all other parameter values.

    apply_torque
        Boolean if torque should affect rotation. If True `omega_u` is ignored and T_ext
        is used as external forcing

    T_ex
        Lambda function for describing the external forcing as a function of time

    outdir
        Directory to put results in

    steps_per_phase
        Number of time steps per phase of the induction engine

    outfile
        File to write results to. (Default is print to terminal)

    plot
        Plot torque and voltage over time

    progress
        Show progress bar for solving in time

    mesh_dir
        Directory containing mesh

    xdmf_file
        Name of XDMF file for output. If None do not write
    """
    freq = model_parameters["freq"]
    T = num_phases * 1 / freq
    dt_ = 1 / steps_per_phase * 1 / freq
    mu_0 = model_parameters["mu_0"]
    omega_J = 2 * np.pi * freq

    ext = "single" if single_phase else "three"
    fname = f"{mesh_dir}/{ext}_phase"

    domains, currents = domain_parameters(single_phase)

    # Read mesh and cell markers
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")

    # Read facet tag
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")

    # Create DG 0 function for mu_R and sigma
    DG0 = dolfinx.FunctionSpace(mesh, ("DG", 0))
    mu_R = dolfinx.Function(DG0)
    sigma = dolfinx.Function(DG0)
    density = dolfinx.Function(DG0)
    for (material, domain) in domains.items():
        for marker in domain:
            cells = ct.indices[ct.values == marker]
            mu_R.x.array[cells] = model_parameters["mu_r"][material]
            sigma.x.array[cells] = model_parameters["sigma"][material]
            density.x.array[cells] = model_parameters["densities"][material]

    # Define problem function space
    cell = mesh.ufl_cell()
    FE = ufl.FiniteElement("Lagrange", cell, degree)
    ME = ufl.MixedElement([FE, FE])
    VQ = dolfinx.FunctionSpace(mesh, ME)

    # Define test, trial and functions for previous timestep
    Az, V = ufl.TrialFunctions(VQ)
    vz, q = ufl.TestFunctions(VQ)
    AnVn = dolfinx.Function(VQ)
    An, _ = ufl.split(AnVn)  # Solution at previous time step
    J0z = dolfinx.Function(DG0)  # Current density

    # Create integration sets
    Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
    Omega_c = domains["Rotor"] + domains["Al"]

    # Create integration measures
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=surface_map["Exterior"])

    # Define temporal and spatial parameters
    n = ufl.FacetNormal(mesh)
    dt = dolfinx.Constant(mesh, dt_)
    x = ufl.SpatialCoordinate(mesh)

    omega = dolfinx.Constant(mesh, omega_u)

    # Define variational form
    a = dt / mu_R * ufl.inner(ufl.grad(Az), ufl.grad(vz)) * dx(Omega_n + Omega_c)
    a += dt / mu_R * vz * (n[0] * Az.dx(0) - n[1] * Az.dx(1)) * ds
    a += mu_0 * sigma * Az * vz * dx(Omega_c)
    a += dt * mu_0 * sigma * (V.dx(0) * q.dx(0) + V.dx(1) * q.dx(1)) * dx(Omega_c)
    L = dt * mu_0 * J0z * vz * dx(Omega_n)
    L += mu_0 * sigma * An * vz * dx(Omega_c)

    # Motion voltage term
    u = omega * ufl.as_vector((-x[1], x[0]))
    a += dt * mu_0 * sigma * ufl.dot(u, ufl.grad(Az)) * vz * dx(Omega_c)

    # Find all dofs in Omega_n for Q-space
    cells_n = np.hstack([ct.indices[ct.values == domain] for domain in Omega_n])
    Q = VQ.sub(1).collapse()
    deac_dofs = dolfinx.fem.locate_dofs_topological((VQ.sub(1), Q), tdim, cells_n)

    # Create zero condition for V in Omega_n
    zeroQ = dolfinx.Function(Q)
    zeroQ.x.array[:] = 0
    bc_Q = dolfinx.DirichletBC(zeroQ, deac_dofs, VQ.sub(1))

    # Create external boundary condition for V space
    V_ = VQ.sub(0).collapse()
    tdim = mesh.topology.dim

    def boundary(x):
        return np.full(x.shape[1], True)

    boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, boundary)
    bndry_dofs = dolfinx.fem.locate_dofs_topological((VQ.sub(0), V_), tdim - 1, boundary_facets)
    zeroV = dolfinx.Function(V_)
    zeroV.x.array[:] = 0
    bc_V = dolfinx.DirichletBC(zeroV, bndry_dofs, VQ.sub(0))
    bcs = [bc_V, bc_Q]

    # Create sparsity pattern and matrix with additional non-zeros on diagonal
    cpp_a = dolfinx.Form(a, form_compiler_parameters=form_compiler_parameters,
                         jit_parameters=jit_parameters)._cpp_object
    pattern = dolfinx.cpp.fem.create_sparsity_pattern(cpp_a)
    block_size = VQ.dofmap.index_map_bs
    deac_blocks = deac_dofs[0] // block_size
    pattern.insert_diagonal(deac_blocks)
    pattern.assemble()

    # Create matrix based on sparsity pattern
    A = dolfinx.cpp.la.create_matrix(mesh.mpi_comm(), pattern)
    A.zeroEntries()
    if not apply_torque:
        A.zeroEntries()
        dolfinx.fem.assemble_matrix(A, cpp_a, bcs=bcs)
        A.assemble()

    # Create inital vector for LHS
    cpp_L = dolfinx.Form(L, form_compiler_parameters=form_compiler_parameters,
                         jit_parameters=jit_parameters)._cpp_object
    b = dolfinx.fem.create_vector(cpp_L)

    # Create solver
    solver = PETSc.KSP().create(mesh.mpi_comm())
    solver.setOperators(A)
    prefix = "AV_"
    solver.setOptionsPrefix(prefix)
    opts = PETSc.Options()
    opts[f"{prefix}ksp_type"] = "preonly"
    opts[f"{prefix}pc_type"] = "lu"
    # opts[f"{prefix}ksp_type"] = "gmres"
    # opts[f"{prefix}pc_type"] = "sor"

    # opts[f"{prefix}ksp_converged_reason"] = None
    # opts[f"{prefix}ksp_monitor_true_residual"] = None
    # opts[f"{prefix}ksp_gmres_modifiedgramschmidt"] = None
    # opts[f"{prefix}ksp_diagonal_scale"] = None
    # opts[f"{prefix}ksp_gmres_restart"] = 500
    # opts[f"{prefix}ksp_rtol"] = 1e-08
    # opts[f"{prefix}ksp_max_it"] = 50000
    # opts[f"{prefix}ksp_view"] = None
    # opts[f"{prefix}ksp_monitor"] = None
    solver.setFromOptions()
    # Function for containg the solution
    AzV = dolfinx.Function(VQ)

    # Post-processing function for projecting the magnetic field potential
    post_B = MagneticFieldProjection2D(AzV)

    # Class for computing torque, losses and induced voltage
    derived = DerivedQuantities2D(AzV, AnVn, u, sigma, domains, ct, ft)

    # Create output file
    if xdmf_file is not None:
        postproc = XDMFWrapper(mesh.mpi_comm(), xdmf_file)
        postproc.write_mesh(mesh)
        # postproc.write_function(sigma, 0, "sigma")
        # postproc.write_function(mu_R, 0, "mu_R")

    # Computations needed for adding addiitonal torque to engine
    x = ufl.SpatialCoordinate(mesh)
    r = ufl.sqrt(x[0]**2 + x[1]**2)
    L = 1  # Depth of domain
    I_rotor = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(L * r**2 * density * dx(Omega_c)))

    # Post proc variables
    torques = [0]
    torques_vol = [0]
    times = [0]
    omegas = [omega_u]
    pec_tot = [0]
    pec_steel = [0]
    VA = [0]
    VmA = [0]
    # Generate initial electric current in copper windings
    t = 0
    update_current_density(J0z, omega_J, t, ct, currents)

    if MPI.COMM_WORLD.rank == 0 and progress:
        progress = tqdm.tqdm(desc="Solving time-dependent problem",
                             total=int(T / float(dt.value)))

    while t < T:
        # Update time step and current density
        if MPI.COMM_WORLD.rank == 0 and progress:
            progress.update(1)
        t += float(dt.value)
        update_current_density(J0z, omega_J, t, ct, currents)

        with dolfinx.common.Timer("~Reassemble LHS") as ttt:
            # Reassemble LHS
            if apply_torque:
                A.zeroEntries()
                dolfinx.fem.assemble_matrix(A, cpp_a, bcs=bcs)
                A.assemble()

        with dolfinx.common.Timer("~Reassemble RHS") as ttt:
            # Reassemble RHS
            with b.localForm() as loc_b:
                loc_b.set(0)
            dolfinx.fem.assemble_vector(b, cpp_L)
            dolfinx.fem.apply_lifting(b, [cpp_a], [bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.set_bc(b, bcs)

        with dolfinx.common.Timer("~Solve problem") as ttt:
            # Solve problem
            solver.solve(b, AzV.vector)
            AzV.x.scatter_forward()

        with dolfinx.common.Timer("~Compute losses, torque and induced voltage") as ttt:
            # Compute losses, torque and induced voltage
            loss_al, loss_steel = derived.compute_loss(float(dt.value))
            pec_tot.append(float(dt.value) * (loss_al + loss_steel))
            pec_steel.append(float(dt.value) * loss_steel)
            torques.append(derived.torque_surface())
            torques_vol.append(derived.torque_volume())
            vA, vmA = derived.compute_voltage(float(dt.value))
            VA.append(vA)
            VmA.append(vmA)
            times.append(t)

        with dolfinx.common.Timer("~Update previous time step") as ttt:
            # Update previous time step
            AnVn.x.array[:] = AzV.x.array
            AnVn.x.scatter_forward()

        # Update rotational speed depending on torque
        if apply_torque:
            omega.value += float(dt.value) * (derived.torque_volume() - T_ext(t)) / I_rotor
            omegas.append(float(omega.value))

        with dolfinx.common.Timer("~Write solution to file") as ttt:
            # Write solution to file
            if xdmf_file is not None:
                # Project B = curl(Az)
                post_B.solve()

                postproc.write_function(AzV.sub(0).collapse(), t, "Az")
                # postproc.write_function(AzV.sub(1).collapse(), t, "V")
                # postproc.write_function(J0z, t, "J0z")
                postproc.write_function(post_B.B, t, "B")

    if xdmf_file is not None:
        postproc.close()

    with dolfinx.common.Timer("~Postproc") as ttt:

        times = np.asarray(times)
        torques = np.asarray(torques)
        torques_vol = np.asarray(torques_vol)
        VA = np.asarray(VA)
        VmA = np.asarray(VmA)
        pec_tot = np.asarray(pec_tot)
        pec_steel = np.asarray(pec_steel)

        # Compute torque and voltage over last period only
        num_periods = np.round(60 * T)
        last_period = np.flatnonzero(np.logical_and(times > (num_periods - 1) / 60, times < num_periods / 60))
        steps = len(last_period)
        VA_p = VA[last_period]
        VmA_p = VmA[last_period]
        min_T, max_T = min(times[last_period]), max(times[last_period])
        torque_v_p = torques_vol[last_period]
        torque_p = torques[last_period]
        avg_torque, avg_vol_torque = np.sum(torque_v_p) / steps, np.sum(torque_p) / steps

        pec_tot_p = np.sum(pec_tot[last_period]) / (max_T - min_T)
        pec_steel_p = np.sum(pec_steel[last_period]) / (max_T - min_T)
        RMS_Voltage = np.sqrt(np.dot(VA_p, VA_p) / steps) + np.sqrt(np.dot(VmA_p, VmA_p) / steps)
        # RMS_T = np.sqrt(np.dot(torque_p, torque_p) / steps)
        # RMS_T_vol = np.sqrt(np.dot(torque_v_p, torque_v_p) / steps)
        elements = mesh.topology.index_map(mesh.topology.dim).size_global
        num_dofs = VQ.dofmap.index_map.size_global * VQ.dofmap.index_map_bs
        # Print values for last period
        if mesh.mpi_comm().rank == 0:
            print(f"{omega_u}, {avg_torque}, {avg_vol_torque}, {RMS_Voltage}, {pec_tot_p}, {pec_steel_p}, "
                  + f"{num_phases}, {steps_per_phase}, {freq}, {degree}, {elements}, {num_dofs}, {single_phase}",
                  file=outfile)
        ttt.stop()
    # Plot over all periods
    if mesh.mpi_comm().rank == 0 and plot:
        plt.figure()
        plt.plot(times, torques, "--r", label="Surface Torque")
        plt.plot(times, torques_vol, "-b", label="Volume Torque")
        plt.plot(times[last_period], torque_v_p, "--g")
        plt.grid()
        plt.legend()
        plt.savefig(f"{outdir}/torque_{omega_u}_{ext}.png")
        if apply_torque:
            plt.figure()
            plt.plot(times, omegas, "-ro", label="Angular velocity")
            plt.title(f"Angular velocity {omega_u}")
            plt.grid()
            plt.legend()
            plt.savefig(f"{outdir}/omega_{omega_u}_{ext}.png")

        plt.figure()
        plt.plot(times, VA, "-ro", label="Phase A")
        plt.plot(times, VmA, "-ro", label="Phase -A")
        plt.title("Induced Voltage in Phase A and -A")
        plt.grid()
        plt.legend()
        plt.savefig(f"{outdir}/voltage_{omega_u}_{ext}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scripts to  solve the TEAM 30 problem"
        + " (http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _single = parser.add_mutually_exclusive_group(required=False)
    _single.add_argument('--single', dest='single', action='store_true',
                         help="Solve single phase problem", default=False)
    _three = parser.add_mutually_exclusive_group(required=False)
    _three.add_argument('--three', dest='three', action='store_true',
                        help="Solve three phase problem", default=False)
    _torque = parser.add_mutually_exclusive_group(required=False)
    _torque.add_argument('--apply-torque', dest='apply_torque', action='store_true',
                         help="Apply external torque to engine (ignore omega)", default=False)
    parser.add_argument("--num_phases", dest='num_phases', type=int, default=6, help="Number of phases to run")
    parser.add_argument("--omega", dest='omegaU', type=np.float64, default=0, help="Angular speed of rotor [rad/s]")
    parser.add_argument("--degree", dest='degree', type=int, default=1,
                        help="Degree of magnetic vector potential functions space")
    parser.add_argument("--steps", dest='steps', type=int, default=100,
                        help="Time steps per phase of the induction engine")
    parser.add_argument("--outdir", dest='outdir', type=str, default=None,
                        help="Directory for results")
    _plot = parser.add_mutually_exclusive_group(required=False)
    _plot.add_argument('--plot', dest='plot', action='store_true',
                       help="Plot induced voltage and torque over time", default=False)
    _plot = parser.add_mutually_exclusive_group(required=False)
    _plot.add_argument('--progress', dest='progress', action='store_true',
                       help="Show progress bar", default=False)

    args = parser.parse_args()

    def T_ext(t):
        T = args.num_phases * 1 / 60
        if t > 0.5 * T:
            return 1
        else:
            return 0

    outdir = args.outdir
    if args.outdir is None:
        outdir = "results"
    os.system(f"mkdir -p {outdir}")
    if args.single:
        xdmf_file = f"{outdir}/TEAM30_{args.omegaU}_single.xdmf"
        solve_team30(True, args.num_phases, args.omegaU, args.degree, apply_torque=args.apply_torque, T_ext=T_ext,
                     outdir=outdir, steps_per_phase=args.steps, plot=args.plot, progress=args.progress,
                     xdmf_file=xdmf_file)
    if args.three:
        xdmf_file = f"{outdir}/TEAM30_{args.omegaU}_three.xdmf"
        solve_team30(False, args.num_phases, args.omegaU, args.degree, apply_torque=args.apply_torque, T_ext=T_ext,
                     outdir=outdir, steps_per_phase=args.steps, plot=args.plot, progress=args.progress,
                     xdmf_file=xdmf_file)

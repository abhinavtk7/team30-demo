"""
Solving a Permanent Magnet Synchronous Motor (PMSM) problem 
using the TEAM-30 code as the foundational framework.
"""

import argparse
import sys
from io import TextIOWrapper
from pathlib import Path
from datetime import datetime
from typing import Optional, TextIO, Union, Dict
import math

import dolfinx.fem.petsc as _petsc
import dolfinx.mesh
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import ufl
from dolfinx import cpp, fem, io, default_scalar_type
from dolfinx.io import VTXWriter
from mpi4py import MPI
from petsc4py import PETSc


from utils import DerivedQuantities2D, MagneticField2D, update_current_density


def solve_pmsm(outdir: Path = Path("results"), plot: bool = False, progress: bool = False, save_output: bool = False):
    """
    Solve the TEAM 30 problem for a single or three phase engine.

    Parameters
    ==========
    outdir
        Directory to put results in

    plot
        Plot torque and voltage over time

    progress
        Show progress bar for solving in time

    save_output
        Save output to bp-files
    """

    # Parameters
    fname = Path("meshes") / "pmesh3"               # pmsm mesh {pmesh3, pmesh1, test4}
    num_phases: int = 6                             # Number of phases to run (default: 6)
    omega_u: np.float64 = 62.83                     # Angular speed of rotor [rad/s]    # 600 RPM; 1 RPM = 2pi/60 rad/s
    degree: np.int32 = 1                            # Degree of magnetic vector potential functions space (default: 1)
    steps_per_phase = 100                           # Time steps per phase of the induction engine (default: 100)
    apply_torque: bool = False                      # Apply external torque to engine (ignore omega) (default: False)
    outfile: Optional[Union[TextIOWrapper, TextIO]] = sys.stdout
    petsc_options: dict = {"ksp_type": "preonly", "pc_type": "lu"}
    form_compiler_options: dict = {} 
    jit_parameters: dict = {}

    # Note: model_parameters, domain_parameters and surface_map imported from generate_pmsm_2D script
    # Model parameters for the PMSM model
    model_parameters = {
        "mu_0": 1.25663753e-6,      # Relative permability of air [H/m]=[kg m/(s^2 A^2)]
        "freq": 50,                 # Frequency of excitation,
        "J": 3.1e6 * np.sqrt(2),    # [A/m^2] Current density of copper winding
        "mu_r": {"Cu": 1, "Stator": 30, "Rotor": 30, "Al": 1, "Air": 1, "AirGap": 1, "PM": 1.04457},        # Relative permability
        "sigma": {"Rotor": 1.6e6, "Al": 3.72e7, "Stator": 0, "Cu": 0, "Air": 0, "AirGap": 0, "PM": 6.25e5},  # Conductivity 6
        "densities": {"Rotor": 7850, "Al": 2700, "Stator": 0, "Air": 0, "Cu": 0, "AirGap": 0, "PM": 7500}       # [kg/m^3]
    }

    freq = model_parameters["freq"]             # 50
    T = num_phases * 1 / freq                   # 0.1
    dt_ = 1 / steps_per_phase * 1 / freq        # 0.00016666666666666666 # steps_per_phase = 100
    mu_0 = model_parameters["mu_0"]             # 1.25663753e-06
    omega_J = 2 * np.pi * freq                  # 376.99111843077515

    # Copper wires and PMs are ordered in counter clock-wise order from angle = 0, 2*np.pi/num_segments...
    domains = {"Air": (1,), "AirGap": (2, 3), "Al": (4,), "Rotor": (5, ), 
               "Stator": (6, ), "Cu": (7, 8, 9, 10, 11, 12),
               "PM": (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)}
    
    # Currents mapping to the domain markers of copper
    currents = {7: {"alpha": 1, "beta": 0}, 8: {"alpha": -1, "beta": 2 * np.pi / 3},
                9: {"alpha": 1, "beta": 4 * np.pi / 3}, 10: {"alpha": -1, "beta": 0},
                11: {"alpha": 1, "beta": 2 * np.pi / 3},
                12: {"alpha": -1, "beta": 4 * np.pi / 3}}

    # Marker for facets, and restriction to use in surface integral of airgap
    surface_map: Dict[str, Union[int, str]] = {"Exterior": 1, "MidAir": 2, "restriction": "+"}

    # Read mesh and cell markers
    with io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()                                 # <dolfinx.mesh.Mesh object at 0x7f556028faf0>
        ct = xdmf.read_meshtags(mesh, name="Cell_markers")      # <dolfinx.mesh.MeshTags object at 0x7fa5de47f730>
        tdim = mesh.topology.dim        # 2
        mesh.topology.create_connectivity(tdim - 1, 0)
        ft = xdmf.read_meshtags(mesh, name="Facet_markers")     # <dolfinx.mesh.MeshTags object at 0x7fa5de47f6a0>
    
    # Create DG 0 function for mu_R and sigma
    DG0 = fem.FunctionSpace(mesh, ("DG", 0))
    mu_R = fem.Function(DG0)
    sigma = fem.Function(DG0)
    density = fem.Function(DG0)
    for (material, domain) in domains.items():
        for marker in domain:
            cells = ct.find(marker)
            mu_R.x.array[cells] = model_parameters["mu_r"][material]        # Add material properties to each cells
            sigma.x.array[cells] = model_parameters["sigma"][material]      
            density.x.array[cells] = model_parameters["densities"][material]

    # Define problem function space
    cell = mesh.ufl_cell()                              
    FE = ufl.FiniteElement("Lagrange", cell, degree)    
    ME = ufl.MixedElement([FE, FE])                     
    VQ = fem.FunctionSpace(mesh, ME)        

    # Define test, trial and functions for previous timestep
    Az, V = ufl.TrialFunctions(VQ)
    vz, q = ufl.TestFunctions(VQ)
    AnVn = fem.Function(VQ)
    An, _ = ufl.split(AnVn)  # Solution at previous time step
    J0z = fem.Function(DG0)  # Current density
    
    # Create integration sets
    Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
    Omega_c = domains["Rotor"] + domains["Al"] + domains["PM"]
    Omega_pm = domains["PM"]
    

    # Magnetization part
    coercivity = 8.38e5  # [A/m]   
    spacing1 = (np.pi / 6) + (np.pi / 30)
    angles1 = np.asarray([i * spacing1 for i in range(10)], dtype=np.float64)
    
    # link pm orientation angle to each marker
    pm_orientation = {}
    for i, pm_marker in enumerate(Omega_pm):
        pm_orientation[pm_marker] = angles1[i]

    # Magnetization part
    DG0v = fem.FunctionSpace(mesh, ("DG", 0, (2,)))
    Mvec = fem.Function(DG0v)
    block_size = 2 # Mvec.function_space.dofmap.index_map_bs = 2 for 2D

    for (material, domain) in domains.items():
        if material == 'PM':
            for marker in domain:
                angle = pm_orientation[marker]
                Mx = coercivity * np.cos(angle)
                My = coercivity * np.sin(angle)

                cells = ct.find(marker)
                for cell in cells:
                    idx = block_size * cell
                    Mvec.x.array[idx + 0] = Mx
                    Mvec.x.array[idx + 1] = My

    # Mvec.x.scatter_forward()
    Mvec.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)


    # Create integration measures
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=surface_map["Exterior"])

    # Define temporal and spatial parameters
    n = ufl.FacetNormal(mesh)
    dt = fem.Constant(mesh, dt_)
    x = ufl.SpatialCoordinate(mesh)

    omega = fem.Constant(mesh, default_scalar_type(omega_u))

    # Motion voltage term
    u = omega * ufl.as_vector((-x[1], x[0]))

    # Magnetization term
    curl_vz = ufl.as_vector((vz.dx(1), -vz.dx(0)))
    mag_term = (mu_0/mu_R) * ufl.inner( Mvec , curl_vz) * dx(Omega_pm)
    
    # Define variational form
    f_a =   + dt / mu_R * ufl.inner(ufl.grad(Az), ufl.grad(vz)) * dx(Omega_n + Omega_c) \
            + mu_0 * sigma * (Az - An) * vz * dx(Omega_c) \
            + dt * mu_0 * sigma * ufl.dot(u, ufl.grad(Az)) * vz * dx(Omega_c) \
            - dt * mu_0 * J0z * vz * dx(Omega_n) \
            - mag_term

    f_v =   + dt * mu_0 * sigma * (V.dx(0) * q.dx(0) + V.dx(1) * q.dx(1)) * dx(Omega_c)

    form_av = f_a + f_v
    a, L = ufl.system(form_av)

    # Find all dofs in Omega_n for Q-space
    cells_n = np.hstack([ct.find(domain) for domain in Omega_n])
    Q, _ = VQ.sub(1).collapse()
    deac_dofs = fem.locate_dofs_topological((VQ.sub(1), Q), tdim, cells_n)

    # Create zero condition for V in Omega_n
    zeroQ = fem.Function(Q)
    zeroQ.x.array[:] = 0
    bc_Q = fem.dirichletbc(zeroQ, deac_dofs, VQ.sub(1))

    # Create external boundary condition for V space
    V_, _ = VQ.sub(0).collapse()
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bndry_dofs = fem.locate_dofs_topological((VQ.sub(0), V_), tdim - 1, boundary_facets)
    zeroV = fem.Function(V_)
    zeroV.x.array[:] = 0
    bc_V = fem.dirichletbc(zeroV, bndry_dofs, VQ.sub(0))
    bcs = [bc_V, bc_Q]

    # Create sparsity pattern and matrix with additional non-zeros on diagonal
    cpp_a = fem.form(a, form_compiler_options=form_compiler_options,
                     jit_options=jit_parameters)
    pattern = fem.create_sparsity_pattern(cpp_a)
    block_size = VQ.dofmap.index_map_bs
    deac_blocks = deac_dofs[0] // block_size
    pattern.insert_diagonal(deac_blocks)
    pattern.finalize()

    # Create matrix based on sparsity pattern
    A = cpp.la.petsc.create_matrix(mesh.comm, pattern)
    A.zeroEntries()
    if not apply_torque:
        A.zeroEntries()
        _petsc.assemble_matrix(A, cpp_a, bcs=bcs)  # type: ignore
        A.assemble()

    # Create inital vector for LHS
    cpp_L = fem.form(L, form_compiler_options=form_compiler_options,
                     jit_options=jit_parameters)
    b = _petsc.create_vector(cpp_L)

    # Create solver
    solver = PETSc.KSP().create(mesh.comm)  # type: ignore
    solver.setOperators(A)
    prefix = "AV_"

    # Give PETSc solver options a unique prefix
    solver_prefix = f"PMSM_solve_{id(solver)}"    
    solver.setOptionsPrefix(solver_prefix)

    # Set PETSc options
    opts = PETSc.Options()  # type: ignore
    opts.prefixPush(solver_prefix)
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    solver.setFromOptions()
    solver.setOptionsPrefix(prefix)
    solver.setFromOptions()

    # Function for containg the solution
    AzV = fem.Function(VQ)
    Az_out = AzV.sub(0).collapse()
    V_out = AzV.sub(1).collapse()

    # Post-processing function for projecting the magnetic field potential
    post_B = MagneticField2D(AzV)

    # Class for computing torque, losses and induced voltage
    derived = DerivedQuantities2D(AzV, AnVn, u, sigma, domains, ct, ft)
    Az_out.name = "Az"
    post_B.B.name = "B"

    V_out.name = "V"

    # Create output file
    if save_output:
        Az_vtx = VTXWriter(mesh.comm, str(outdir / "Az.bp"), [Az_out])
        B_vtx = VTXWriter(mesh.comm, str(outdir / "B.bp"), [post_B.B])
        V_vtx = VTXWriter(mesh.comm, str(outdir / "V.bp"), [V_out])
    # Computations needed for adding addiitonal torque to engine
    x = ufl.SpatialCoordinate(mesh)
    r = ufl.sqrt(x[0]**2 + x[1]**2)
    L = 1  # Depth of domain
    I_rotor = mesh.comm.allreduce(fem.assemble_scalar(fem.form(L * r**2 * density * dx(Omega_c))))

    # Post proc variables
    num_steps = int(T / float(dt.value))
    torques = np.zeros(num_steps + 1, dtype=default_scalar_type)
    torques_vol = np.zeros(num_steps + 1, dtype=default_scalar_type)
    times = np.zeros(num_steps + 1, dtype=default_scalar_type)
    omegas = np.zeros(num_steps + 1, dtype=default_scalar_type)
    omegas[0] = omega_u
    pec_tot = np.zeros(num_steps + 1, dtype=default_scalar_type)
    pec_steel = np.zeros(num_steps + 1, dtype=default_scalar_type)
    VA = np.zeros(num_steps + 1, dtype=default_scalar_type)
    VmA = np.zeros(num_steps + 1, dtype=default_scalar_type)
    # Generate initial electric current in copper windings
    t = 0.
    update_current_density(J0z, omega_J, t, ct, currents)

    if MPI.COMM_WORLD.rank == 0 and progress:
        progressbar = tqdm.tqdm(desc="Solving time-dependent problem",
                                total=int(T / float(dt.value)))

    for i in range(num_steps):
        # Update time step and current density
        if MPI.COMM_WORLD.rank == 0 and progress:
            progressbar.update(1)
        t += float(dt.value)
        update_current_density(J0z, omega_J, t, ct, currents)


        # Reassemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        _petsc.assemble_vector(b, cpp_L)
        _petsc.apply_lifting(b, [cpp_a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        fem.set_bc(b, bcs)

        # Solve problem
        solver.solve(b, AzV.vector)
        AzV.x.scatter_forward()
    
        # Compute losses, torque and induced voltage
        loss_al, loss_steel = derived.compute_loss(float(dt.value))
        pec_tot[i + 1] = float(dt.value) * (loss_al + loss_steel)
        pec_steel[i + 1] = float(dt.value) * loss_steel
        torques[i + 1] = derived.torque_surface()
        torques_vol[i + 1] = derived.torque_volume()
        vA, vmA = derived.compute_voltage(float(dt.value))
        VA[i + 1] = vA
        VmA[i + 1] = vmA
        times[i + 1] = t

        # Update previous time step
        AnVn.x.array[:] = AzV.x.array
        AnVn.x.scatter_forward()

        # Update rotational speed 
        omegas[i + 1] = float(omega.value)

        # Write solution to file
        if save_output:
            post_B.interpolate()
            Az_out.x.array[:] = AzV.sub(0).collapse().x.array[:]
            V_out.x.array[:] = AzV.sub(1).collapse().x.array[:]
            Az_vtx.write(t)
            B_vtx.write(t)
            V_vtx.write(t)
    b.destroy()

    if save_output:
        Az_vtx.close()
        B_vtx.close()
        V_vtx.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to  solve the PMSM problem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plot', dest='plot', action='store_true',
                        help="Plot induced voltage and torque over time", default=False)
    parser.add_argument('--progress', dest='progress', action='store_true',
                        help="Show progress bar", default=False)
    parser.add_argument('--output', dest='output', action='store_true',
                        help="Save output to VTXFiles files", default=False)

    args = parser.parse_args()

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%b_%d_%H_%M_%S")
    outdir = Path(f"PMSM_{formatted_datetime}")
    outdir.mkdir(exist_ok=True)
    solve_pmsm(outdir=outdir, plot=args.plot, progress=args.progress, save_output=args.output)

import argparse
import os

import dolfinx
import dolfinx.io
import numpy as np
import tqdm
import ufl
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from generate_team30_meshes import r2, r3
# Model parameters
mu_0 = 1.25663753e-6  # Relative permability of air
freq = 60  # Frequency of excitation
omega_J = 2 * np.pi * freq
J = 3.1e6 * np.sqrt(2)  # [A/m^2] Current density of copper winding

_mu_r = {"Cu": 1, "Stator": 30, "Rotor": 30, "Al": 1, "Air": 1}
_sigma = {"Rotor": 1.6e6, "Al": 3.72e7, "Stator": 0, "Cu": 0, "Air": 0}

# Single phase model domains:
# Copper (0 degrees): 1
# Copper (180 degrees): 2
# Steel Stator: 3
# Steel rotor: 4
# Air: 5, 6, 8, 9, 10
# Alu rotor: 7
_domains_single = {"Cu": (1, 2), "Stator": (3,), "Rotor": (4,),
                   "Al": (7,), "Air": (5, 6, 8, 9, 10)}
# Currents on the form J_0 = (0,0, alpha*J*cos(omega*t + beta)) in domain i
_currents_single = {1: {"alpha": 1, "beta": 0}, 2: {"alpha": -1, "beta": 0}}
# Domain data for air gap between rotor and windings, MidAir is the tag an internal interface
# AirGap is the domain markers for the domain
_torque_single = {"MidAir": (11,), "restriction": "+", "AirGap": (5, 6)}
# Three phase model domains:
# Copper (0 degrees): 1
# Copper (60 degrees): 2
# Copper (120 degrees): 3
# Copper (180 degrees): 4
# Copper (240 degrees): 5
# Copper (300 degrees): 6
# Steel Stator: 7
# Steel rotor: 8
# Air: 9, 10, 12, 13, 14, 15, 16, 17, 18
# Alu rotor: 11
_domains_three = {"Cu": (1, 2, 3, 4, 5, 6), "Stator": (7,), "Rotor": (8,),
                  "Al": (10,), "Air": (9, 11, 12, 13, 14, 15, 16, 17, 18)}
# Domain data for air gap between rotor and windings, MidAir is the tag an internal interface
# AirGap is the domain markers for the domain
_torque_three = {"MidAir": (19,), "restriction": "+", "AirGap": (9, 10)}
# Currents on the form J_0 = (0,0, alpha*J*cos(omega*t + beta)) in domain i
_currents_three = {1: {"alpha": 1, "beta": 0}, 2: {"alpha": -1, "beta": 2 * np.pi / 3},
                   3: {"alpha": 1, "beta": 4 * np.pi / 3}, 4: {"alpha": -1, "beta": 0},
                   5: {"alpha": 1, "beta": 2 * np.pi / 3}, 6: {"alpha": -1, "beta": 4 * np.pi / 3}}


def cross_2D(A, B):
    return A[0] * B[1] - A[1] * B[0]


class PostProcessing(dolfinx.io.XDMFFile):
    """
    Post processing class adding a sligth overhead to the XDMFFile class
    """

    def __init__(self, comm: MPI.Intracomm, filename: str):
        super(PostProcessing, self).__init__(comm, f"{filename}.xdmf", "w")

    def write_function(self, u, t, name: str = None):
        if name is not None:
            u.name = name
        super(PostProcessing, self).write_function(u, t)


def update_current_density(J_0, omega, t, ct, currents):
    """
    Given a DG-0 scalar field J_0, update it to be alpha*J*cos(omega*t + beta)
    in the domains with copper windings
    """
    J_0.x.array[:] = 0
    for domain, values in currents.items():
        _cells = ct.indices[ct.values == domain]
        J_0.x.array[_cells] = np.full(len(_cells), J * values["alpha"]
                                      * np.cos(omega * t + values["beta"]))


def solve_team30(single_phase: bool, T: np.float64, omega_u: np.float64, degree: np.int32,
                 form_compiler_parameters: dict = {}, jit_parameters: dict = {}):
    """
    Solve the TEAM 30 problem for a single or three phase engine.
      Parameters
    ==========
    single_phase
        If true run the single phase model, otherwise run the three phase model
    T
        End time of simulation
    omega_u
        Angular speed of rotor
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
    """
    dt_ = 0.05 / omega_J  # FIXME: Add control over dt

    ext = "single" if single_phase else "three"
    fname = f"meshes/{ext}_phase"

    if single_phase:
        domains = _domains_single
        currents = _currents_single
        torque_data = _torque_single
    else:
        domains = _domains_three
        currents = _currents_three
        torque_data = _torque_three

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
    for (material, domain) in domains.items():
        for marker in domain:
            cells = ct.indices[ct.values == marker]
            mu_R.x.array[cells] = _mu_r[material]
            sigma.x.array[cells] = _sigma[material]

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
    Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"]
    Omega_c = domains["Rotor"] + domains["Al"]

    # Create integration measures
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=mesh)

    # Define temporal and spatial parameters
    n = ufl.FacetNormal(mesh)
    dt = dolfinx.Constant(mesh, dt_)
    x = ufl.SpatialCoordinate(mesh)
    r = ufl.sqrt(x[0]**2 + x[1]**2)
    omega = dolfinx.Constant(mesh, omega_u)

    # Define variational form
    a = dt / mu_R * ufl.inner(ufl.grad(Az), ufl.grad(vz)) * dx(Omega_n + Omega_c)
    a += dt / mu_R * vz * (n[0] * Az.dx(0) - n[1] * Az.dx(1)) * ds
    a += mu_0 * sigma * Az * vz * dx(Omega_c)
    a += dt * mu_0 * sigma * (V.dx(0) * q.dx(0) + V.dx(1) * q.dx(1)) * dx(Omega_c)
    L = dt * mu_0 * J0z * vz * dx(Omega_n)
    L += mu_0 * sigma * An * vz * dx(Omega_c)

    # Motion voltage term
    u = omega * r * ufl.as_vector((-x[1] / r, x[0] / r))
    a += dt * mu_0 * sigma * ufl.dot(u, ufl.grad(Az)) * vz * dx(Omega_c)

    # Find all dofs in Omega_n for Q-space
    cells_n = np.hstack([ct.indices[ct.values == domain] for domain in Omega_n])
    Q = VQ.sub(1).collapse()
    deac_dofs = dolfinx.fem.locate_dofs_topological((VQ.sub(1), Q), tdim, cells_n)

    # Create zero condition for V in Omega_n
    zeroQ = dolfinx.Function(Q)
    zeroQ.x.array[:] = 0
    bc_Q = dolfinx.DirichletBC(zeroQ, deac_dofs, VQ.sub(1))

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

    # Assemble matrix once as it is time-independent
    bcs = [bc_V, bc_Q]
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

    AzV = dolfinx.Function(VQ)

    # Create output file
    postproc = PostProcessing(mesh.mpi_comm(), f"results/TEAM30_{omega_u}_{ext}")
    postproc.write_mesh(mesh)
    # postproc.write_function(sigma, 0, "sigma")
    # postproc.write_function(mu_R, 0, "mu_R")

    # Create variational form for electromagnetic field B
    el_B = ufl.VectorElement("DG", cell, degree - 1)
    VB = dolfinx.FunctionSpace(mesh, el_B)
    ub = ufl.TrialFunction(VB)
    vb = ufl.TestFunction(VB)
    aB = ufl.inner(ub, vb) * ufl.dx
    _Az, _ = ufl.split(AzV)
    LB = ufl.inner(ufl.as_vector((_Az.dx(1), _Az.dx(0))), vb) * ufl.dx
    cpp_aB = dolfinx.fem.Form(aB, form_compiler_parameters=form_compiler_parameters, jit_parameters=jit_parameters)
    cpp_LB = dolfinx.fem.Form(LB, form_compiler_parameters=form_compiler_parameters, jit_parameters=jit_parameters)
    AB = dolfinx.fem.assemble_matrix(cpp_aB)
    bB = dolfinx.fem.create_vector(cpp_LB)
    AB.assemble()
    B = dolfinx.Function(VB)
    solverB = PETSc.KSP().create(mesh.mpi_comm())
    solverB.setOperators(AB)
    prefixB = "B_"
    solverB.setOptionsPrefix(prefixB)
    # opts[f"{prefixB}ksp_monitor"] = None
    opts[f"{prefixB}ksp_type"] = "preonly"
    opts[f"{prefixB}pc_type"] = "lu"
    solverB.setFromOptions()

    # Create variational form for Electromagnetic torque
    Brst = B(torque_data["restriction"])
    dS_air = ufl.Measure("dS", domain=mesh, subdomain_data=ft, subdomain_id=torque_data["MidAir"])
    L = 1
    dF = 1 / mu_0 * (ufl.dot(Brst, x / r) * Brst - 0.5 * ufl.dot(Brst, Brst) * x / r)
    # NOTE: Fake integration over dx to orient normals
    torque = L * cross_2D(x, dF) * dS_air + dolfinx.Constant(mesh, 0) * dx(0)

    # Volume formulation of torque
    # https://www.comsol.com/blogs/how-to-analyze-an-induction-motor-a-team-benchmark-model/
    dx_gap = ufl.Measure("dx", domain=mesh, subdomain_data=ct, subdomain_id=torque_data["AirGap"])
    Bphi = ufl.inner(B, ufl.as_vector((-x[1], x[0]))) / r
    Br = ufl.inner(B, x) / r
    torque_vol = (r * L / (mu_0 * (r3 - r2)) * Br * Bphi) * dx_gap

    # Generate initial electric current in copper windings
    t = 0
    update_current_density(J0z, omega_J, t, ct, currents)

    progress = tqdm.tqdm(desc="Solving time-dependent problem",
                         total=int(T / float(dt.value)))
    torques = []
    torques_vol = []
    times = []
    while t < T:
        # Update time step and current density
        progress.update(1)
        t += float(dt.value)
        update_current_density(J0z, omega_J, t, ct, currents)

        # Reassemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        dolfinx.fem.assemble_vector(b, cpp_L)
        dolfinx.fem.apply_lifting(b, [cpp_a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, bcs)

        # Solve problem
        solver.solve(b, AzV.vector)
        AzV.x.scatter_forward()

        # Update solution at previous time step
        with AzV.vector.localForm() as loc, AnVn.vector.localForm() as loc_n:
            loc.copy(result=loc_n)
        AnVn.x.scatter_forward()

        # Create vector field B
        with bB.localForm() as loc_b:
            loc_b.set(0)
        dolfinx.fem.assemble_vector(bB, cpp_LB)
        bB.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        solverB.solve(bB, B.vector)
        B.x.scatter_forward()

        # Assemble torque
        T_k = dolfinx.fem.assemble_scalar(torque)
        T_k_vol = dolfinx.fem.assemble_scalar(torque_vol)
        torques.append(mesh.mpi_comm().allreduce(T_k, op=MPI.SUM))
        torques_vol.append(mesh.mpi_comm().allreduce(T_k_vol, op=MPI.SUM))
        times.append(t)

        # Write solution to file
        postproc.write_function(AzV.sub(0).collapse(), t, "Az")
        postproc.write_function(AzV.sub(1).collapse(), t, "V")
        postproc.write_function(J0z, t, "J0z")
        postproc.write_function(B, t, "B")
    postproc.close()

    if mesh.mpi_comm().rank == 0:
        plt.plot(times, torques, "-ro", label="Surface Torque")
        plt.plot(times, torques_vol, "-bs", label="Volume Torque")
        plt.grid()
        plt.legend()
        torques = np.asarray(torques)
        torques_vol = np.asarray(torques_vol)

        RMS_T = np.sqrt(np.dot(torques, torques) / len(times))
        RMS_T_vol = np.sqrt(np.dot(torques_vol, torques_vol) / len(times))
        print(f"RMS Torque: {RMS_T}")
        print(f"RMS Torque Vol: {RMS_T_vol}")
        plt.savefig(f"results/torque_{omega_u}_{ext}.png")


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
    parser.add_argument("--T", dest='T', type=np.float64, default=0.1, help="End time of simulation")
    parser.add_argument("--omega", dest='omegaU', type=np.float64, default=0, help="Angular speed of rotor")
    parser.add_argument("--degree", dest='degree', type=np.int32, default=1,
                        help="Degree of magnetic vector potential functions space")
    args = parser.parse_args()

    os.system("mkdir -p results")
    if args.single:
        solve_team30(True, args.T, args.omegaU, args.degree)
    if args.three:
        solve_team30(False, args.T, args.omegaU, args.degree)

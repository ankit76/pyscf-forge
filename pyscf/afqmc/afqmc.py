import os
import pickle
from functools import partial
from typing import Union

import numpy as np
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from pyscf import __config__, scf
from pyscf.afqmc import config, utils

print = partial(print, flush=True)


class AFQMC:
    def __init__(
        self, mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD]
    ):
        self.mf_or_cc = mf_or_cc
        self.basis_coeff = None
        self.norb_frozen = 0
        self.chol_cut = 1e-5
        self.integrals = None  # custom integrals
        self.mpi_prefix = None
        self.nproc = 1
        self.script = None
        self.dt = 0.005
        self.n_walkers = 50
        self.n_prop_steps = 50
        self.n_ene_blocks = 1
        self.n_sr_blocks = 5
        self.n_blocks = 200
        self.n_ene_blocks_eql = 1
        self.n_sr_blocks_eql = 5
        self.seed = np.random.randint(1, int(1e6))
        self.n_eql = 20
        self.ad_mode = None
        self.orbital_rotation = True
        self.do_sr = True
        self.walker_type = "rhf"
        self.symmetry = False
        self.save_walkers = False
        if isinstance(mf_or_cc, scf.uhf.UHF) or isinstance(mf_or_cc, scf.rohf.ROHF):
            self.trial = "uhf"
        elif isinstance(mf_or_cc, scf.rhf.RHF):
            self.trial = "rhf"
        elif isinstance(mf_or_cc, UCCSD):
            self.trial = "ucisd"
        elif isinstance(mf_or_cc, CCSD):
            self.trial = "cisd"
        else:
            self.trial = None
        self.ene0 = 0.0
        self.n_batch = 1
        self.tmpdir = __config__.TMPDIR + f"/afqmc{np.random.randint(1, int(1e6))}/"

    def kernel(self):
        os.system(f"mkdir -p {self.tmpdir}")
        utils.prep_afqmc(
            self.mf_or_cc,
            basis_coeff=self.basis_coeff,
            norb_frozen=self.norb_frozen,
            chol_cut=self.chol_cut,
            integrals=self.integrals,
            tmpdir=self.tmpdir,
        )
        options = {}
        for attr in dir(self):
            if (
                attr
                not in [
                    "mf_or_cc",
                    "basis_coeff",
                    "norb_frozen",
                    "chol_cut",
                    "integrals",
                    "mpi_prefix",
                    "nproc",
                    "script",
                ]
                and not attr.startswith("__")
                and not callable(getattr(self, attr))
            ):
                options[attr] = getattr(self, attr)
        return run_afqmc(options, self.script, self.mpi_prefix, self.nproc, self.tmpdir)


def run_afqmc(options=None, script=None, mpi_prefix=None, nproc=None, tmpdir="."):
    if options is None:
        options = {}
    with open(tmpdir + "/options.bin", "wb") as f:
        pickle.dump(options, f)
    if script is None:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        script = f"{dir_path}/mpi_jax.py"
    use_gpu = config.afqmc_config["use_gpu"]
    use_mpi = config.afqmc_config["use_mpi"]

    if not use_gpu and config.afqmc_config["use_mpi"] is not False:
        try:
            from mpi4py import MPI

            MPI.Finalize()
            use_mpi = True
            print(f"# mpi4py found, using MPI.")
            if nproc is None:
                print(f"# Number of MPI ranks not specified, using 1 by default.")
        except ImportError:
            use_mpi = False
            print(f"# Unable to import mpi4py, not using MPI.")
        # use_mpi = False
    gpu_flag = "--use_gpu" if use_gpu else ""
    mpi_flag = "--use_mpi" if use_mpi else ""
    if mpi_prefix is None:
        if use_mpi:
            mpi_prefix = "mpirun "
            if nproc is not None:
                mpi_prefix += f"-np {nproc} "

        else:
            mpi_prefix = ""
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script} {tmpdir} {gpu_flag} {mpi_flag}"
    )

    try:
        ene_err = np.loadtxt(tmpdir + "/ene_err.txt")
    except:
        print("AFQMC did not execute correctly.")
        ene_err = 0.0, 0.0
    return ene_err[0], ene_err[1]


def run_afqmc_fp(options=None, script=None, mpi_prefix=None, nproc=None):
    if options is None:
        options = {}
    with open("options.bin", "wb") as f:
        pickle.dump(options, f)
    if script is None:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        script = f"{dir_path}/mpi_jax.py"
    if mpi_prefix is None:
        mpi_prefix = "mpirun "
    if nproc is not None:
        mpi_prefix += f"-np {nproc} "
    os.system(
        f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {mpi_prefix} python {script}"
    )
    # ene_err = np.loadtxt('ene_err.txt')
    # return ene_err[0], ene_err[1]

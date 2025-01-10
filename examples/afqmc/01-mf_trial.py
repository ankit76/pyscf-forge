from pyscf import afqmc, gto, scf

# AFQMC with RHF trial
mol = gto.M(atom="H 0 0 0; H 0 0 2; H 0 0 4; H 0 0 6", basis="sto-3g", unit="Bohr")
mf = scf.RHF(mol)
mf.kernel()

myafqmc = afqmc.AFQMC(mf)
myafqmc.seed = 285055
myafqmc.kernel()

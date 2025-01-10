from pyscf import afqmc, cc, gto, scf

# AFQMC with CISD trial
mol = gto.M(atom="H 0 0 0; H 0 0 2; H 0 0 4; H 0 0 6", basis="sto-3g", unit="Bohr")
mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

myafqmc = afqmc.AFQMC(mycc)
myafqmc.seed = 418108
myafqmc.kernel()

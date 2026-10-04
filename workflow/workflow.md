1. Use boltz to predict protein structre as pdb
2. Add Hydrogens using PDBfixed
3. Convert this to non periodic .xyz file using ase 
4. Use MACE models to calculate the binding energies: E(A.P)-E(A)-E(P)
[P is peptite, A is allene]
https://github.com/ACEsuit/mace-foundations/releases#release-mace_omol_0
To begin with use the MH-1 models [OMOL too big]
5. Plot against stability (half life) and calculate spearman coefficient

LATER
?. Calculate MACE descriptors on peptide 
    ?a. One could remove all atoms with r_eff of model to do this to save cost, but may not be suitable for an electrostatic model?
?. 
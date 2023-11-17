import pandas as pd
import numpy as np
from rdkit.Chem.SaltRemover import SaltRemover
from IPython.display import clear_output
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs, Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs
def standardization(smiles):
    # Code borrowed from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ
    mol = Chem.MolFromSmiles(smiles)
    while True:
        try:
            if mol != None:
            # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
                clean_mol = rdMolStandardize.Cleanup(mol) 

                #remover = SaltRemover() # use default saltremover
                #clean_mol = remover.StripMol(clean_mol)
                # if many fragments, get the "parent" (the actual mol we are interested in) 
                parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
                # try to neutralize molecule
                uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
                uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
                # note that no attempt is made at reionization at this step
                # nor at ionization at some pH (rdkit has no pKa caculator)
                # the main aim to to represent all molecules from different sources
                # in a (single) standard way, for use in ML, catalogue, etc.
                te = rdMolStandardize.TautomerEnumerator() # idem
                taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
            else:
                taut_uncharged_parent_clean_mol = None
            break
        except:
            taut_uncharged_parent_clean_mol = None
            break
    return taut_uncharged_parent_clean_mol
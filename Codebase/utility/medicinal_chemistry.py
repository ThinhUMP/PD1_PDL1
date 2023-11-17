from pathlib import Path
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, PandasTools
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Draw import IPythonConsole
from tqdm.auto import tqdm
tqdm.pandas()


def calculate_ro5_properties(mol, fullfill = 3):
    """
    Test if input molecule (SMILES) fulfills Lipinski's rule of five.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.
    fullfill: int
        Number of rules fullfill RO5

    Returns
    -------
    bool
        Lipinski's rule of five compliance for input molecule.
    """
    # RDKit molecule from SMILES
    if mol != None:
        molecule = mol
        # Calculate Ro5-relevant chemical properties
        molecular_weight = Descriptors.ExactMolWt(molecule)
        n_hba = Descriptors.NumHAcceptors(molecule)
        n_hbd = Descriptors.NumHDonors(molecule)
        logp = Descriptors.MolLogP(molecule)
        #tpsa = Descriptors.TPSA(molecule)
        # Check if Ro5 conditions fulfilled
        conditions = [molecular_weight <= 500, n_hba <= 10, n_hbd <= 5, logp <= 5]
        ro5_fulfilled = sum(conditions) >= fullfill
    else:
        ro5_fulfilled = False
    return ro5_fulfilled

from scopy.ScoFH import fh_filter

def pains_filter(mol):
    """
    PAINS filter for an input molecule (SMILES).

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    [bool, list, list]
        [pains_accepted, pains_matched_name, pains_matched_atoms]
        Check if PAINS not violated and matched names, atoms.
    """
    if mol !=None:
    # RDKit molecule from SMILES
        molecule = Chem.AddHs(mol)
        # Check PAINS
        pains = fh_filter.Check_PAINS(molecule, detail = True)
        pains_accepted = pains['Disposed'] == 'Accepted' # Return True if not violating PAINS
        pains_matched_atoms = pains['MatchedAtoms']
        pains_matched_names = pains['MatchedNames']
        # Return PAINS
    else:
        pains_accepted = False
    return pains_accepted
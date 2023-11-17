from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, PandasTools, AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import cDataStructs
import os
import seaborn as sns
sns.set(style ='darkgrid')
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    PandasTools,
    Draw,
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,)
from map4 import MAP4Calculator
m4_calc = MAP4Calculator(is_folded=True)
from rdkit.Avalon import pyAvalonTools as fpAvalon

class similarity_calculate:
    """
    Calculate molecular fingerprints and Similarity
    
    Input:
    -----
    data : pandas.DataFrame
        Data with SMILES and target columns
    query : rdkit.Chem.rdchem.Mol
        Base molecules
    smile_col: str
        Name of SMILES columns 
    acitivity_col: str
        Name of acitivity columns (pIC50, pChEMBL Value)
    save_name: str 
        Keyword for saved files
    save_dir: str
        Location saving Image
   
    Returns:
    --------
    data: pandas.DataFrame
        Data contains similarity coeficient
    figure:
        Similarity distribution 
    """
    def __init__(self, data, query, smile_col, active_col,save_dir, save_name = 'simi_', ):
        self.data = data
        self.query = query
        self.smile_col = smile_col
        self.active_col = active_col
        self.save_name = save_name
        self.save_dir = save_dir
        PandasTools.AddMoleculeColumnToFrame(self.data, smilesCol = self.smile_col)
        
    def convert_arr2vec(self, arr):
        arr_tostring = "".join(arr.astype(str))
        arr_tostring
        EBitVect2 = cDataStructs.CreateFromBitString(arr_tostring)
        return EBitVect2
        
    def fingerprint(self):
        #query
        self.maccs_query = MACCSkeys.GenMACCSKeys(self.query)
        self.avalon_query = fpAvalon.GetAvalonFP(self.query, 1024) 
        self.ecfp4_query = AllChem.GetMorganFingerprintAsBitVect(self.query, 2, nBits=2048)
        self.rdk5_query = Chem.RDKFingerprint(self.query, maxPath=5, fpSize=2048, nBitsPerHash=2)
        self.map4_query = m4_calc.calculate(self.query)
        self.map4_query_vec = self.convert_arr2vec(self.map4_query)
      
        #list
        self.maccs_list = self.data["ROMol"].apply(MACCSkeys.GenMACCSKeys).tolist()
        self.avalon_list = self.data["ROMol"].apply(fpAvalon.GetAvalonFP, nBits=1024).tolist()
        self.ecfp4_list = self.data["ROMol"].apply(AllChem.GetMorganFingerprintAsBitVect, radius= 2, nBits=2048).tolist()
        self.rdk5_list = self.data["ROMol"].apply(Chem.RDKFingerprint, maxPath=5, fpSize=2048, nBitsPerHash=2).tolist() 
        self.map4_list = self.data["ROMol"].apply(m4_calc.calculate)
        self.map4_list_vec = self.map4_list.apply(self.convert_arr2vec)
    
    def Coef(self):
        # Tanimoto
        self.data["tanimoto_avalon"] = DataStructs.BulkTanimotoSimilarity(self.avalon_query, self.avalon_list)
        self.data["tanimoto_maccs"] = DataStructs.BulkTanimotoSimilarity(self.maccs_query, self.maccs_list)  
        self.data["tanimoto_ecfp4"] = DataStructs.BulkTanimotoSimilarity(self.ecfp4_query, self.ecfp4_list)
        self.data["tanimoto_rdk5"] = DataStructs.BulkTanimotoSimilarity(self.rdk5_query, self.rdk5_list)  
        self.data["tanimoto_map4"] = DataStructs.BulkTanimotoSimilarity(self.map4_query_vec, self.map4_list_vec)

        # Dice
        self.data["dice_avalon"] = DataStructs.BulkDiceSimilarity(self.avalon_query, self.avalon_list)
        self.data["dice_maccs"] = DataStructs.BulkDiceSimilarity(self.maccs_query, self.maccs_list)
        self.data["dice_ecfp4"] = DataStructs.BulkDiceSimilarity(self.ecfp4_query, self.ecfp4_list)
        self.data["dice_rdk5"] = DataStructs.BulkDiceSimilarity(self.rdk5_query, self.rdk5_list)
        self.data["dice_map4"] = DataStructs.BulkTanimotoSimilarity(self.map4_query_vec, self.map4_list_vec)

        
   
        
    def fit(self):
        self.fingerprint()
        self.Coef()
        
        self.tani_col = []
        self.dice_col  = []
        for key, values in enumerate(self.data.columns):
            if 'tanimoto' in values:
                self.tani_col.append(values)
            elif 'dice' in values:
                self.dice_col.append(values)
        display(self.data.head(5))
        self.data.to_csv(f"{self.save_dir}/Raw_data/{self.save_name}{self.query.GetProp('_Name')}.csv")
        
    def plot(self):
        for i in range(len(self.tani_col)):
            fig, axes = plt.subplots(figsize=(14, 6), nrows=1, ncols=2)
            sns.histplot(data = self.data, x=self.data[self.tani_col[i]],hue=self.data[self.active_col], ax=axes[0], kde = True, )
            sns.histplot(data = self.data, x=self.data[self.dice_col[i]],hue=self.data[self.active_col], ax=axes[1], kde = True, )
            plt.legend(loc='lower left', title='Molecular fingerprints')
            fig.savefig(f"{self.save_dir}/Image/{self.query.GetProp('_Name')}_{self.tani_col[i][9:]}.png", transparent = True, dpi = 600)
            plt.show()
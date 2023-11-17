import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import Chem, DataStructs
from mhfp.encoder import MHFPEncoder
class featurizer:
    def __init__(self, data, ID_col, smiles_col, type_fp = 'RDK7', active_col = 'pChEMBL Value'):
        self.data = data
        self.ID_col = ID_col
        self.smiles_col = smiles_col
        self.type_fp = type_fp
        self.active_col = active_col
        self.data['Molecule'] = self.data[self.smiles_col].apply(Chem.MolFromSmiles)
    
    def RDKFp(self, mol, maxPath=7, fpSize=4096, nBitsPerHash=2):
        fp = Chem.RDKFingerprint(mol, maxPath=maxPath, fpSize=fpSize, nBitsPerHash=nBitsPerHash)
        ar = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, ar)
        return ar
    
    def fit(self):
        if self.type_fp == 'RDK7':
            self.data['FPs'] = self.data['Molecule'].apply(self.RDKFp)
            X = np.stack(self.data.FPs.values)
            df = pd.DataFrame(X)
            df= pd.concat([self.data, df], axis = 1).drop([self.data.columns[1],"FPs", "Molecule"], axis =1)
            
        if self.type_fp == 'secfp':
            self.data["FPs"] = self.data[self.smiles_col].apply(MHFPEncoder.secfp_from_smiles)
            X = np.stack(self.data.FPs.values)
            df = pd.DataFrame(X)
            df= pd.concat([self.data, df], axis = 1).drop([self.data.columns[1],"FPs", "Molecule"], axis =1)
        
        if self.active_col in df.columns:
            df[self.active_col] =df[self.active_col]
        else:
            df[self.active_col] = 0
        df.columns = df.columns.astype('string')
        return df
   
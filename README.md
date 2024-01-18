# PDL1

# **Innovative Virtual Screening of PD-L1 Inhibitors: The Synergy of Molecular Similarity, Neural Networks, and GNINA Docking**

![The workflow of the study](https://github.com/ThinhUMP/PD1_PDL1/blob/main/Image/Result/Artboard%201.png)

## **Abstract**
Immune checkpoint inhibitors are popular cancer treatment researchs over the years. Many studies indicated that PD-L1 inhibitors prevent cancer cells from avoiding the immune system by reactivating tumor cell death program through T cell. This study aimed to develop a neural network model (ANN), molecular similarity (MS) and GNINA 1.0 molecular docking model to screen PD-L1 inhibitors. Database of 2044 substances with PD-L1 inhibitory activity were collected from Google Patents, then used in molecular similarity, and machine learning model. This study employed hPD-L1 protein (PDB ID: 5N2F) for the retrospective control of the docking procedure. Subsequently, 15235 compounds from the Drugbank database underwent screening through medicinal chemistry filters, MS, the ANN model. and finally, molecular docking - GNINA 1.0. The decoy generation achieved promising results, with AUC-ROC 1NN of 0.52, Doppelganger scores mean of 0.24, and Doppelganger scores max of 0.346, indicating that the decoys closely resemble the active set. In MS establishment, the AVALON fingerprint was the best nominee for similarity searching, with EF1% of 10.96%, AUC-ROC of 0.963, and a similarity threshold of 0.32. The ANN model attained an average precision of 0.863±0.032 and F1 score of 0.745±0.039 in cross-validation, higher than those of the Support Vector Classifier (SVC) and Random Forest (RF) models, although without a significant difference. In external evaluation, the ANN model exhibited an average precision of 0.851 and F1 score of 0.790, also higher than those of the SVC and RF models. GNINA 1.0 was performed and validated by redocking for docking power and retrospective control for screening power, with the AUC metrics being 0.975 and the threshold being identified based on a cnn_pose_score of 0.73. Finally, 128 compounds from DrugBank repurposing data were selected from the MS and ANN models, then 22 candidates were identified as potential compounds from GNINA 1.0, and (3S)-1-(4-acetylphenyl)-5-oxopyrrolidine-3-carboxylic acid was detected as the most promising molecule, with cnn_pose_score of 0.79, PD-L1 inhibitory probability: 70.5%, and aTanimoto coefficient of 0.35.

![The results of virtual screening process](https://github.com/ThinhUMP/PD1_PDL1/blob/main/Image/Result/Virtual%20screening%20results.png)

**KEYWORDS**: Drug discovery, GNINA, molecular similarity, PD-L1 inhibitors, virtual screening.

# Installation

This project used a `setup.py` script for installation. Here are the steps to install it:

1. Clone the repository: git clone https://github.com/ThinhUMP/PD1_PDL1.git
2. Navigate into the cloned project directory
3. Install the project using `setup.py`: python setup.py install

# Contributors
- [Van-Thinh To](https://thinhump.github.io/)
- [Phuoc-Chung Nguyen Van](https://www.facebook.com/chung.nguyenvanphuoc.9)
- [Tieu-Long Phan](https://tieulongphan.github.io/)
- [Tuyen Ngoc Truong](https://scholar.google.com/citations?hl=vi&user=qx3eMsIAAAAJ) - [Corresponding author](mailto:truongtuyen@ump.edu.vn)

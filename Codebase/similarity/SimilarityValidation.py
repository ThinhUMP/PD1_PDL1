import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, precision_score,  roc_curve, auc, precision_recall_curve,average_precision_score, f1_score
from sklearn.metrics import accuracy_score, make_scorer
import matplotlib.pyplot as plt
class similarity_validation:
       
    """
    Similarity Validation.

    Parameters
    ----------
    data : pandas.DataFrame
        Data after post processing with "Active", "predict" and "rescore" columns.
    active : str
        Name of "Active" column (binary).
    model : str
        Identification of Model.     
    scores: float
        Docking score, RMSD in pharamacophore searching or rescore columns.

    Returns
    -------
    table: pandas.DataFrame
        Data with validation metrics: Model-Sensitivity-Specificity-AUCROC-logAUCROC-BedROC-EF1%-RIE.
    plot: matplot
        ROC plot
        
    """
    
    def __init__(self, data, active_col, query, save_dir,scores = 'tanimoto',
                 plot_type = 'roc', figsize = (14,10)):
        self.data = data
        self.active_col = active_col
        self.query = query
        self.scores = scores  
        self.plot_type = plot_type
        self.figsize = figsize
        self.save_dir = save_dir
        if self.figsize == None:
            pass
        else:
            fig = plt.figure(figsize = self.figsize)
            background_color = "#F0F6FC"
            fig.patch.set_facecolor(background_color)
        sns.set()
        
        self.simi_col = []
        for key, values in enumerate(self.data.columns):
            if self.scores in values:
                self.simi_col.append(values)    
       
    def EF(self, actives_list, score_list, n_percent):
        """ Calculates enrichment factor.
        Parameters:
        actives_list - binary array of active/decoy status.
        score_list - array of experimental scores.
        n_percent - a decimal percentage.
        """
        total_actives = len(actives_list[actives_list == 1])
        total_compounds = len(actives_list)
        # Sort scores, while keeping track of active/decoy status
        # NOTE: This will be inefficient for large arrays
        labeled_hits = sorted(zip(score_list, actives_list), reverse=True)
        # Get top n percent of hits
        num_top = int(total_compounds * n_percent)
        top_hits = labeled_hits[0:num_top]    
        num_actives_top = len([value for score, value in top_hits if value == 1])
        # Calculate enrichment factor
        return num_actives_top / (total_actives * n_percent)
    
    def roc_log_auc(self, y_true, y_score, pos_label=None, ascending_score=True,
                    log_min=0.001, log_max=1.):
        """Computes area under semi-log ROC.
        Parameters
        ----------
        y_true : array, shape=[n_samples]
            True binary labels, in range {0,1} or {-1,1}. If positive label is
            different than 1, it must be explicitly defined.
        y_score : array, shape=[n_samples]
            Scores for tested series of samples
        pos_label: int
            Positive label of samples (if other than 1)
        ascending_score: bool (default=True)
            Indicates if your score is ascendig. Ascending score icreases with
            deacreasing activity. In other words it ascends on ranking list
            (where actives are on top).
        log_min : float (default=0.001)
            Minimum value for estimating AUC. Lower values will be clipped for
            numerical stability.
        log_max : float (default=1.)
            Maximum value for estimating AUC. Higher values will be ignored.
        Returns
        -------
        auc : float
            semi-log ROC AUC
        """
        if ascending_score:
            y_score = -y_score
        fpr, tpr, t = roc_curve(y_true, y_score)
        fpr = fpr.clip(log_min)
        idx = (fpr <= log_max)
        log_fpr = 1 - np.log10(fpr[idx]) / np.log10(log_min)
        return auc(log_fpr, tpr[idx])
    
    def GH_score(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        GH = (0.75*precision + 0.25*recall)*specificity
        return GH


    
    def plot_roc(self,fpr, tpr, thresh, auc, model, base_name):
        """ Calculates and plots and ROC and AUC.
        Parameters:
        actives_list - binary array of active/decoy status.
        score_list - array of experimental scores.
        """
        # Plot figure
        #sns.set('notebook', 'whitegrid', 'dark', font_scale=1.5, font='Ricty',
        #rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
        #plt.figure(figsize = (12,8))
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        cutoff = thresh[ix]
        
        lw = 2
        
        plt.plot(fpr, tpr, 
                 lw=lw, label=f'{model} (AUC = %0.3f), Cutoff = %.2f' % (auc, cutoff))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize = 16)
        plt.ylabel('True Positive Rate', fontsize = 16)
        plt.title(f'ROC curve - {base_name}', fontsize = 24, weight = 'semibold')
        plt.legend(loc="lower right")
        
    def plot_ap(self,precision, recall, thresh, ap, model, base_name):
        """ Calculates and plots PR curve.
        Parameters:
        actives_list - binary array of active/decoy status.
        score_list - array of experimental scores.
        """
        
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        cutoff = thresh[ix]
        
        lw = 2
        plt.plot(recall, precision, 
                 lw=lw, label=f'{model} (AP = %0.3f), Cutoff = %.2f' % (ap, cutoff))
        plt.plot([0, 0], [0, 0], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Precision', fontsize = 16)
        plt.ylabel('Recall', fontsize = 16)
        plt.title(f'PR curve - {base_name}', fontsize = 24, weight = 'semibold')
        plt.legend(loc="lower right")
    
    def validation(self):
        self.model = []
        self.list_roc_auc = []
        self.list_ap = []
        self.list_log_roc_auc = []
        self.list_ef1 = []
        self.list_ef5 = []
        self.list_ef10 = []
        self.list_f1 = []
        self.list_GH = []
        
        for i in self.simi_col:
            self.model.append(i)
            fpr, tpr, _ = roc_curve(self.data[self.active_col], self.data[i])
            roc_auc = round(auc(fpr, tpr),3)
            self.list_roc_auc.append(roc_auc)
            precision, recall, thresholds = precision_recall_curve(self.data[self.active_col], self.data[i])
            ap = round(average_precision_score(self.data[self.active_col], self.data[i]),3)
            self.list_ap.append(ap)
            log_roc_auc = round(self.roc_log_auc(self.data[self.active_col], self.data[i], ascending_score = False),3)
            self.list_log_roc_auc.append(log_roc_auc)
            ef1 = round(self.EF(self.data[self.active_col], self.data[i], 0.01),3)
            self.list_ef1.append(ef1)
            ef5 = round(self.EF(self.data[self.active_col], self.data[i], 0.05),3)
            self.list_ef5.append(ef5)
            ef10 = round(self.EF(self.data[self.active_col], self.data[i], 0.1),3)
            self.list_ef10.append(ef10)
            #f1 = round(f1_score(self.data[self.active_col], self.data[i]),3)
            #self.list_f1.append(f1)
            #gh = round(self.GH_score(self.data[self.active_col], self.data[i]),3)
            #self.list_GH.append(gh)
            
            
        index = ['Model', "AP", "AUCROC", "logAUCROC",
                 "EF1%","EF5%", "EF10%"] #,'F1', 'GH'
        metric =[self.model,self.list_ap, self.list_roc_auc, self.list_log_roc_auc, 
                 self.list_ef1, self.list_ef5, self.list_ef10] #self.list_f1,self.list_GH
        self.table = pd.DataFrame(data = metric, index = index).T
        self.table.to_csv(f"{self.save_dir}/Raw_data/Validation_{self.query.GetProp('_Name')}.csv")
        
    def visualize(self):
        if self.plot_type == 'roc':
            sns.set()
            plt.figure(figsize = self.figsize)
            name = self.query.GetProp('_Name')
            for i in self.simi_col:
                fpr, tpr, thresh = roc_curve(self.data[self.active_col], self.data[i])
                roc_auc = round(auc(fpr, tpr),3)
                self.plot_roc(fpr, tpr, thresh, roc_auc,model = i, base_name = name)
            plt.savefig(f'{self.save_dir}/Image/{self.plot_type }_{name}.png', dpi = 600)
                
        elif self.plot_type == 'pr':
            sns.set()
            plt.figure(figsize = self.figsize)
            name = self.query.GetProp('_Name')
            for i in self.simi_col:
                precision, recall, thresholds = precision_recall_curve(self.data[self.active_col], self.data[i])
                ap = round(average_precision_score(self.data[self.active_col], self.data[i]),3)
                self.plot_ap(precision, recall, thresholds, ap, model = i, base_name = self.query.GetProp('_Name'))
                plt.savefig(f'{self.save_dir}/Image/{self.plot_type }_{name}.png', dpi = 600)
            
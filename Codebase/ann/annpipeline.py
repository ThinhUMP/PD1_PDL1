from sklearn.metrics import f1_score, average_precision_score, classification_report
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.utils.data import TensorDataset
import random
import numpy as np
import torch

from sklearn.metrics import f1_score, average_precision_score, classification_report



class model_pipeline():
    import random
    import numpy as np
    import torch
    def __init__(self,device, seed, save_dir):
        self.device = device
        self.seed = seed
        self.save_dir = save_dir



    def seed_everything(self):

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def train(self):
        self.model.train()
        trainning_loss = 0
        truelabels = []
        probas = []
        proba_flat = []
        pred_flat = []
        for fps, labels in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(fps)   
            
            loss = self.criterion(output[:,0], labels)
            
             # # Specify L1 and L2 weights
            # # l1_weight = 0.3
            l2_weight = 0.0001

            # Compute L1 and L2 loss component
            parameters = []
            for parameter in self.model.parameters():
                parameters.append(parameter.view(-1))
            # l1 = l1_weight * model.compute_l1_loss(torch.cat(parameters))
            l2 = l2_weight * self.model.compute_l2_loss(torch.cat(parameters))

            # # Add L1 and L2 loss components
            # # loss += l1
            loss += l2
            
            
            loss.backward()
            self.optimizer.step()
            trainning_loss += loss.item()
            
           

            probas.append(np.asarray(output.detach().cpu()))
            a = list(np.asarray(labels.detach().cpu()))
            truelabels.append(a)

        for i in probas:
            for j in i:
                proba_flat.append(j)

        pred_flat = proba_flat.copy()
        for key, value in enumerate(proba_flat):
            if value < 0.5:
                pred_flat[key] = 0
            else:
                pred_flat[key] = 1
        flatten_list = lambda truelabels:[element for item in truelabels for element in flatten_list(item)] if type(truelabels) is list else [truelabels]
        truelabels = flatten_list(truelabels)
        loss = trainning_loss/len(self.train_loader)
        f1 = f1_score(truelabels,pred_flat)
        ap = average_precision_score(truelabels,proba_flat)
        return loss, f1, ap

    def evaluate(self):
        self.model.eval()
        validation_loss = 0
        truelabels_val = []
        probas_val = []
        proba_flat_val = []
        pred_flat_val = []
        with torch.no_grad():
            for fps, labels in self.valid_loader:
                out_put = self.model(fps)
                validation_loss += self.criterion(out_put[:,0], labels).cpu()

                probas_val.append(np.asarray(out_put.detach().cpu()))
                a = list(np.asarray(labels.detach().cpu()))
                truelabels_val.append(a)


        for i in probas_val:
            for j in i:
                proba_flat_val.append(j)

        pred_flat_val = proba_flat_val.copy()
        for key, value in enumerate(proba_flat_val):
            if value < 0.5:
                pred_flat_val[key] = 0
            else:
                pred_flat_val[key] = 1


        flatten_list = lambda truelabels_val:[element for item in truelabels_val for element in flatten_list(item)] if type(truelabels_val) is list else [truelabels_val]
        truelabels_val = flatten_list(truelabels_val)
        loss_val = validation_loss/len(self.valid_loader)
        f1_val = f1_score(truelabels_val,pred_flat_val)
        ap_val = average_precision_score(truelabels_val,proba_flat_val)
        return loss_val, f1_val, ap_val


    def save_model(self, epochs, model, optimizer, criterion,save_dir):
        print("Saving...")
        torch.save({
                'epoch': self.epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.criterion,
                }, self.save_dir + '/ANN_model.pth')

    def fit(self, data_train, data_valid, epochs, show_progress, model, lr=0.001, weight_decay=0.05, factor=0.99, patience=10):
        self.seed_everything()
        self.architecture = model
        self.model = model.to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.lr, weight_decay = self.weight_decay)
        self.criterion = torch.nn.BCELoss()
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.factor, 
                                                                       patience=self.patience, verbose=True)
        self.train_loader = data_train
        self.valid_loader = data_valid
        self.history = {"train_loss":[], "val_loss": [],
              "train_f1":[], "val_f1":[],
              "train_ap":[], "val_ap":[]}
        for epoch in range(self.epochs):
            train1_loss, train1_f1, train1_ap = self.train()

            val1_loss,val1_f1, val1_ap = self.evaluate()

            # self.lr_scheduler.step(train1_f1)
            self.history["train_loss"].append(train1_loss)
            self.history["val_loss"].append(val1_loss.detach().numpy())
            self.history["train_f1"].append(train1_f1)
            self.history["val_f1"].append(val1_f1)
            self.history["train_ap"].append(train1_ap)
            self.history["val_ap"].append(val1_ap)
            if show_progress == True:
                if (epoch+1) % 5 == 0:
                    print("Epoch: {}/{}.. ".format(epoch+1, self.epochs),
              "Training Loss: {:.3f}.. ".format(train1_loss),
              "validation Loss: {:.3f}.. ".format(val1_loss),
            "validation f1_score: {:.3f}.. ".format(val1_f1),
                  "validation average precision: {:.3f}.. ".format(val1_ap),
             )
            else:
                pass


        if show_progress == True:
            self.visualize()
        else:
            pass
        self.save_model(self.epochs, self.model, self.optimizer, self.criterion,self.save_dir)
        print("Complete the process...")

    def predict(self, data_test, checkpoint, model):
        checkpoint = checkpoint
        model = model.to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        model.eval()
        validation_loss = 0
        truelabels_val = []
        probas_val = []
        proba_flat_val = []
        pred_flat_val = []
        with torch.no_grad():
            for fps, labels in data_test:
                out_put = model(fps)
                # validation_loss += self.criterion(out_put[:,0], labels).cpu()

                a = list(np.asarray(labels.detach().cpu()))
                truelabels_val.append(a)

                probas_val.append(np.asarray(out_put.detach().cpu()))


        flatten_list = lambda truelabels_val:[element for item in truelabels_val for element in flatten_list(item)] if type(truelabels_val) is list else [truelabels_val]
        truelabels_val = flatten_list(truelabels_val)
        for i in probas_val:
            for j in i:
                proba_flat_val.append(j)

        pred_flat_val = proba_flat_val.copy()
        for key, value in enumerate(proba_flat_val):
            if value < 0.5:
                pred_flat_val[key] = 0
            else:
                pred_flat_val[key] = 1
        return pred_flat_val, truelabels_val

    def predict_proba(self,data_test, checkpoint, model, criterion):
        checkpoint = checkpoint
        model = model.to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        validation_loss = 0
        truelabels_val = []
        probas_val = []
        proba_flat_val = []
        with torch.no_grad():
            for fps, labels in data_test:
                out_put = model(fps)
                validation_loss += criterion(out_put[:,0], labels).cpu()

                a = list(np.asarray(labels.detach().cpu()))
                truelabels_val.append(a)

                probas_val.append(np.asarray(out_put.detach().cpu()))

        flatten_list = lambda truelabels_val:[element for item in truelabels_val for element in flatten_list(item)] if type(truelabels_val) is list else [truelabels_val]
        truelabels_val = flatten_list(truelabels_val)
        for i in probas_val:
            for j in i:
                proba_flat_val.append(j)

        pred_flat_val = proba_flat_val.copy()
        for key, value in enumerate(proba_flat_val):
            if value < 0.5:
                pred_flat_val[key] = 0
            else:
                pred_flat_val[key] = 1
        return proba_flat_val,truelabels_val

    def visualize(self):
        
        sns.set()

        # create a subplot with three axes
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))


        ax1.plot(self.history["train_loss"], label='Training Loss')
        ax1.plot(self.history["val_loss"], label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss over Epochs')

        ax2.plot(self.history["train_f1"], label='Training F1 Score')
        ax2.plot(self.history["val_f1"],label="Validation F1 Score")
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('F1 score')
        ax2.legend()
        ax2.set_title('F1 Score over Epochs')

        ax3.plot(self.history["train_ap"], label='Training AP Score')
        ax3.plot(self.history["val_ap"],label="Validation AP Score")
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Average precision score')
        ax3.legend()
        ax3.set_title('Average precision Score over Epochs')

        fig.suptitle('Training Metrics over Epochs')


        plt.show()

    def reset_weights(self,model):

        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


    def cross_val_score(self, model,epochs, X_train, y_train, cv):
        self.History = {"F1_record":[],"AP_record":[]}
        for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
            Xtrain = torch.tensor(X_train.iloc[train_index,:].values , device=self.device).float()
            Xtest = torch.tensor(X_train.iloc[test_index,:].values, device=self.device).float()

            ytrain = torch.tensor(y_train.iloc[train_index].values , device=self.device).float()
            ytest = torch.tensor(y_train.iloc[test_index].values, device=self.device).float()

            train_dataset = TensorDataset(Xtrain, ytrain)
            test_dataset = TensorDataset(Xtest, ytest)

            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=32,
                                          shuffle=True)
            self.valid_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=32,
                                              shuffle=False)

            self.model = model.to(self.device)
            self.model.apply(lambda m: self.reset_weights(m))

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr= 0.0001, weight_decay = 0.05)
            self.criterion = torch.nn.BCELoss()
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            for epoch in range(epochs):
                train1_loss,train1_f1, train1_ap = self.train()
                self.lr_scheduler.step(train1_f1)
            val1_loss,val1_f1, val1_ap = self.evaluate()
            print("Fold: {}.. ".format(i+1),
            "validation f1_score: {:.3f}.. ".format(val1_f1),
                  "validation average precision: {:.3f}.. ".format(val1_ap),
             )
            self.History["F1_record"].append(val1_f1)
            self.History["AP_record"].append(val1_ap)
        #if scoring == "average_precision":
        self.mean_scores_ap = sum(self.History["AP_record"]) / len(self.History["AP_record"])
        print(f"Overall AP score = {self.mean_scores_ap :.4f}")
        #if scoring == "f1":
        self.mean_scores_f1 = sum(self.History["F1_record"]) / len(self.History["F1_record"])
        print(f"Overall F1 score = {self.mean_scores_f1:.4f}")
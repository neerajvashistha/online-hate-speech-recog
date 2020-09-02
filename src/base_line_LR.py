from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
import joblib

class LR():
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    
    def train(self, X, y, max_iter, test_size, param_grid, path):
        """
        Trains a logistic regression model, expects feature vector in X, suitable test split in test_size 
        param_grid for paramters and path to save the model
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,shuffle=True, random_state=42, test_size=test_size, stratify=y)
        pipe = Pipeline(
                [('select', SelectFromModel(LogisticRegression(class_weight='balanced',
                                                          penalty="l2", C=0.01, max_iter=max_iter ))),
                ('model', LogisticRegression(class_weight='balanced',penalty='l2', max_iter=max_iter ))])

        param_grid = param_grid # Optionally add parameters here
        grid_search = GridSearchCV(pipe, 
                           param_grid,
                           cv=StratifiedKFold(n_splits=5, 
                                              random_state=42, 
                                              shuffle=True).split(self.X_train, self.y_train), 
                           verbose=2,
                           n_jobs=-1
                          )
        model = grid_search.fit(self.X_train, self.y_train)
        print("Writing Model to file")
        joblib.dump(model.best_estimator_, path)
        print("Done!")
        return model
    
    def predict(self,model, X_test=None):
        """
        makes prediction on X_test
        """
        if X_test is None:
            X_test = self.X_test
        y_pred = model.predict(X_test)
        return y_pred
    
    def gen_report(self, y_test = None, y_pred = None):
        """
        generate accuracy and classification report with Precision, recall, f1 scores 
        """
        if y_test is None:
            y_test = self.y_test
        acc = accuracy_score(y_test, y_pred)
        return acc, classification_report(y_test, y_pred)
    
    def gen_confusion_matrix(self, y_test = None, y_pred = None, classes = 3):
        """
        generates confusion matrix, for 2, 3 class data
        """
        if y_test is None:
            y_test = self.y_test
        confusion_mat = confusion_matrix(y_test, y_pred)
        
        if classes ==3:
            matrix_proportions = np.zeros((3,3))
            for i in range(0,3):
                matrix_proportions[i,:] = confusion_mat[i,:]/float(confusion_mat[i,:].sum())
            names=['Hate','Offensive','Neither']
            confusion_df = pd.DataFrame(matrix_proportions, index=names,columns=names)
            plt.figure(figsize=(5,5))
            seaborn.heatmap(confusion_df,annot=True,annot_kws={"size": 12},cmap='gist_gray_r',cbar=False, square=True,fmt='.2f')
            plt.ylabel(r'True categories',fontsize=14)
            plt.xlabel(r'Predicted categories',fontsize=14)
            plt.tick_params(labelsize=12)
        if classes == 2:
            matrix_proportions = np.zeros((2,2))
            for i in range(0,2):
                matrix_proportions[i,:] = confusion_mat[i,:]/float(confusion_mat[i,:].sum())
            names=['Hate','Neither']
            confusion_df = pd.DataFrame(matrix_proportions, index=names,columns=names)
            plt.figure(figsize=(5,5))
            seaborn.heatmap(confusion_df,annot=True,annot_kws={"size": 12},cmap='gist_gray_r',cbar=False, square=True,fmt='.2f')
            plt.ylabel(r'True categories',fontsize=14)
            plt.xlabel(r'Predicted categories',fontsize=14)
            plt.tick_params(labelsize=12)
    
    def load_model(self,path):
        print('Loading Model from file')
        return joblib.load(path)
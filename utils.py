from IPython.display import display 
import pandas as pd 
import numpy as np 
import os 
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import MinMaxScaler 
class ml_dataset: 
    ## this is used to get a dataset as a class in order to achieve basic manipulation 
    def __init__(self, dataframe, target_variable):
        # first, initiate a dataframe with certain target variable 
        self.dataset = dataframe 
        self.target_variable = target_variable 
        
        df_summary = pd.DataFrame({'PctNull': np.round(self.dataset.isna().sum()/len(self.dataset), 2), 
                              'DataTypes': self.dataset.dtypes})

        # after this step, remember continuous vs. cateogorical variables 
        self.continuous_variables = df_summary[df_summary.DataTypes != 'object'].index
        self.categorical_variables = df_summary[df_summary.DataTypes == 'object'].index

        self.df_summary = df_summary 
        
    def continuous_summary(self): 
        """
        summarize the basic of continuous variables 
        """

        distribution = self.dataset.describe().T 

        continuous_summary = pd.merge(self.df_summary.loc[self.continuous_variables, :], distribution, how='inner', left_index=True, right_index=True) 

        display(continuous_summary)

    def correlation_matrix(self): 
        
        df_correlation = self.dataset[self.continuous_variables].corr('spearman') 

        display(df_correlation) 

    def categorical_summary(self): 
        for var in self.categorical_variables: 
            display(self.dataset.groupby(var).agg({self.target_variable: ['count', 'mean']}).reset_index())

    def summary(self): 
        print('---------------------- Basic Attributes -------------------------')
        display(self.df_summary) 
        print('----------------- Distribution of Continuous Variable -------------------') 
        self.continuous_summary() 
        print('----------------- Corrlation matrix -----------------------------------') 
        self.correlation_matrix() 
        print('----------------- Summary of Categorical Variable ---------------------')
        self.categorical_summary() 

    def fill_null(self, config = {'loan_int_rate': 'mean', 'person_emp_length': 'indicator'}): 
        """
        This is the function to fill na based on certain config 
        config: an input dictionary where we specify different ways to fill na 
        there can be a few ways: 
            - mean: simply fill na with mean, apply with continuous variable 
            - zero: fill na with 0 
            - indicator: fill na with mean, but create another indicator saying this is null 
        """
        for var in config.keys(): 
            method = config[var] 
            if method == 'mean': 
                self.dataset[var] = self.dataset[var].fillna(self.dataset[var].mean())
                                         
            elif method == 'zero': 
                self.dataset[var] = self.dataset[var].fillna(0)

            elif method == 'indicator': 
                self.dataset[var+'_missing'] = self.dataset[var].isna()
                self.dataset[var] = self.dataset[var].fillna(self.dataset[var].mean())
                
        display(self.df_summary) 
        
    def training_prep(self, split = {'train': 0.6, 'test': 0.2, 'validation': 0.2}, target = 'loan_status'): 
        """
        This function will automatically do: 1) Onehot encoding; 2) Train-Validation-Test Split 
        it will return train, test, validation split 
        
        """
        # for categorical variables, perform on hot encoding 
        oneHot = pd.get_dummies(self.dataset[self.categorical_variables], drop_first = True)

        # for continuous variable, perform MinMaxscalar as KNN may be sensitive to scaling of continuous factors 
        scalar = MinMaxScaler() 
        continuous_scaled = pd.DataFrame(scalar.fit_transform(self.dataset[self.continuous_variables]), columns = self.continuous_variables)
        df_prepped = pd.concat([continuous_scaled, oneHot], axis=1)

        y = df_prepped[target].astype('category')
        X = df_prepped.drop(target, axis=1) 

        train_size = split['train'] 
        test_size = split['test'] 
        validation_size = split['validation'] 

        # split out validation first 
        X_train_test, X_validate, y_train_test, y_validate = train_test_split(X, y, test_size=validation_size, random_state=42)

        # then split out train/test 
        X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=test_size/(train_size + test_size), random_state=42)
        
        
        return X_train, X_test, X_validate, y_train, y_test, y_validate

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, roc_auc_score

class classificationModel: 
     def __init__(self, model):
         """
         Initiate model object 
         """
         if model == 'DecisionTree': 
             self.model = DecisionTreeClassifier() 
         elif model == 'KNN': 
             self.model = KNeighborsClassifier() 
         elif model == 'SVM': 
             self.model = SVC(probability=True) 
            
        
     def fit(self, X_train, y_train): 
         """
         Function to fit the model 
         """
         self.model.fit(X_train, y_train) 

     def evaluate_performance(self, X_train, y_train, X_test, y_test, X_validate, y_validate):  
         """
         Evaluate model performance in terms of accuracy and AUC 
         """
         sample = [] 
         accuracy = [] 
         auc = [] 
         
         sample.append('Train') 
         accuracy.append(self.model.score(X_train, y_train)) 
         auc.append(roc_auc_score(y_train, self.model.predict_proba(X_train)[:, 1]))
         
         sample.append('Test') 
         accuracy.append(self.model.score(X_test, y_test))
         auc.append(roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1]))
         
         sample.append('Validate') 
         accuracy.append(self.model.score(X_validate, y_validate)) 
         auc.append(roc_auc_score(y_validate, self.model.predict_proba(X_validate)[:, 1]))
         
         self.performance = pd.DataFrame({'sample': sample, 'accuracy': accuracy, 'auc': auc})
        
         return self.performance 

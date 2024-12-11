from IPython.display import display 
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

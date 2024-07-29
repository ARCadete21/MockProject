import pandas as pd
import numpy as np
import re
from typing import Tuple, Union, Optional
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.model_selection import StratifiedKFold
from scipy import stats


##### DATA PREPROCESSING

# DATA TYPES
def transform_variables_to_boolean(train: pd.DataFrame) -> pd.DataFrame: # test: pd.DataFrame = None
    """
    Convert columns with exactly two unique non-null values to boolean type in the provided DataFrames.

    Parameters:
    - train (pd.DataFrame): The training DataFrame.
    # - test (pd.DataFrame, optional): The test DataFrame. Default is None.

    Returns:
    - pd.DataFrame: The modified training DataFrame.
    # - pd.DataFrame or None: The modified test DataFrame, or None if no test DataFrame was provided.
    """
    # Iterate through columns in train data
    for col in train.columns:
        # Get unique non-null values in the column
        unique_values = train[col].dropna().unique()
        # Count the number of unique values
        n_unique_values = len(unique_values)

        # If there are only two unique values, convert the column to boolean type
        if (n_unique_values == 2) & (train[col].dtype != bool):
            
            valid_sets = ({-1, 0}, {-1, 1}, {0, 1})

            if set(unique_values) in valid_sets:
                largest, smallest = max(unique_values), min(unique_values)
                value_to_boolean = {largest: True, smallest: False}

            elif not(np.issubdtype(train[col].dtype, np.number)):
                value_to_boolean = {unique_values[0]: True, unique_values[1]: False}
            
            else:
                continue

            train[col] = train[col].map(value_to_boolean)

            # If test data is provided, convert the column to boolean type in test data as well
            # if test is not None:
            #     test[col] = test[col].astype(bool)

    return train #, test


def datatype_distinction(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Distinguishes between the numerical and categorical columns in a DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame.

    Returns:
    --------
    numerical : pandas.DataFrame
        DataFrame containing only numerical columns.

    categorical : pandas.DataFrame
        DataFrame containing only categorical columns.
    '''
    # Select numerical columns using select_dtypes with np.number
    numerical = data.select_dtypes(include=np.number).copy()
    
    # Select categorical columns by excluding numerical types
    categorical = data.select_dtypes(exclude=[np.number, 'datetime64[ns]']).copy()
    
    return numerical, categorical


# DATA TRANSFORMATION
def transformation(technique: Union[TransformerMixin], data: pd.DataFrame, 
                   column_transformer: bool = False) -> pd.DataFrame:
    '''
    Applies the specified transformation technique to the DataFrame.

    Parameters:
    -----------
    technique : object
        The transformation technique (e.g., from Scikit-learn) to be applied.

    data : pandas.DataFrame
        The input DataFrame to be transformed.

    column_transformer : bool, optional (default=False)
        Flag to indicate if a column transformer is used for custom column names.

    Returns:
    --------
    data_transformed : pandas.DataFrame
        Transformed DataFrame.

    Notes:
    ------
    - If column_transformer is False, the columns in the transformed DataFrame
      will retain the original column names.
    - If column_transformer is True, the method assumes that technique has a
      get_feature_names_out() method and uses it to get feature names for the
      transformed data, otherwise retains the original column names.
    '''
    # Apply the specified transformation technique to the data
    data_transformed = technique.transform(data)
    
    # Create a DataFrame from the transformed data
    data_transformed = pd.DataFrame(
        data_transformed,
        index=data.index,
        columns=technique.get_feature_names_out() if column_transformer else data.columns
    )
    
    return data_transformed


def data_transform(technique: Union[TransformerMixin], X_train: pd.DataFrame, X_val: Optional[pd.DataFrame] = None,
                    column_transformer: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    '''
    Fits a data transformation technique on the training data and applies the transformation 
    to both the training and validation data.

    Parameters:
    -----------
    technique : object
        The data transformation technique (e.g., from Scikit-learn) to be applied.

    X_train : pandas.DataFrame or array-like
        The training data to fit the transformation technique and transform.

    X_val : pandas.DataFrame or array-like, optional (default=None)
        The validation data to be transformed.

    column_transformer : bool, optional (default=False)
        Flag to indicate if a column transformer is used for custom column names.

    Returns:
    --------
    X_train_transformed : pandas.DataFrame
        Transformed training data.

    X_val_transformed : pandas.DataFrame or None
        Transformed validation data. None if X_val is None.

    Notes:
    ------
    - Fits the transformation technique on the training data (X_train).
    - Applies the fitted transformation to X_train and optionally to X_val if provided.
    '''
    # Fit the transformation technique on the training data
    technique.fit(X_train)
    
    # Apply transformation to the training data
    X_train_transformed = transformation(technique, X_train, column_transformer)
    
    # Apply transformation to the validation data if provided
    X_val_transformed = None
    if X_val is not None:
        X_val_transformed = transformation(technique, X_val, column_transformer)
        
    return X_train_transformed, X_val_transformed


# MISSING VALUES
def drop_missing_values(ax: int, train: pd.DataFrame, test: Optional[pd.DataFrame] = None, drop_perc: float = 50.0
                        ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Drops columns or rows from the train DataFrame and optionally the test DataFrame
    based on the percentage of missing values.
    
    Parameters:
    - ax (int): Axis along which to calculate missing values. 0 for columns, 1 for rows.
                Valid values are 0 or 1.
    - train (pd.DataFrame): Training dataset.
    - test (Optional[pd.DataFrame]): Test dataset. Default is None.
    - drop_perc (float): Threshold percentage for dropping columns/rows. Valid values are between 0 and 100.

    Returns:
    - Tuple[pd.DataFrame, Optional[pd.DataFrame]]: Modified train and test DataFrames.
    """
    # Calculate the number of missing values along the specified axis
    axis_nulls = train.isnull().sum(axis=ax)

    # Calculate the size of the train data along the specified axis
    if ax == 0:
        size = len(train.index)
    else:
        size = len(train.columns)

    # Calculate the percentage of missing values
    nulls_percentage = round(100 * axis_nulls / size, 1)
    
    # Initialize a list to store columns with high missing percentage
    to_drop = []
    count = 0
    
    # Print columns to remove
    # print('REMOVE')
    for obj, perc in nulls_percentage.items():
        if perc > drop_perc:
            # print(f'{obj}: {perc}%')
            to_drop.append(obj)
            count += 1
    
    # Remove columns with high missing percentage
    train.drop(to_drop, axis=abs(ax-1), inplace=True)
    
    # Remove the same columns from the test data if ax is 0
    if test is not None and (ax == 0):
        test.drop(to_drop, axis=abs(ax-1), inplace=True)

    # print('Total:', count)

    return train, test



# ENCODING
def one_hot_encoding(X_train: pd.DataFrame) -> pd.DataFrame:
    # Filter the dataset with only the object data type columns
    X_train_obj = X_train.select_dtypes(include=['object'])

    # Get the number of unique values from the filtered dataset
    X_train_obj_nu = X_train_obj.nunique()

    # Get the columns with more than 2 unique values
    columns_to_encode = X_train_obj_nu.index[X_train_obj_nu > 2]

    # One-Hot
    ct = ColumnTransformer([
        ('oneHot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), columns_to_encode)
        ], remainder='passthrough')
    X_train = data_transform(ct, X_train, column_transformer=True)[0]
    X_train.columns = X_train.columns.str.replace(r'(oneHot|remainder)__', '', regex=True)

    # Tranform variables with only two unique value to boolean
    X_train = transform_variables_to_boolean(X_train)

    return X_train


##### FEATURE ENGINEERING

# FEATURE CREATION
def calculate_mean_difference_inner(data, grouped_variable, by_variables, new_column, means):
    # Merge data with mean values based on 'by_variables'
    merged_data = data.merge(means, on=by_variables, how='left')
    
    # Calculate mean difference
    merged_data[new_column] = merged_data[grouped_variable] - merged_data['mean_' + grouped_variable]
    
    # Drop redundant columns
    return merged_data.drop(columns=['mean_' + grouped_variable])


def calculate_mean_difference(train, grouped_variable, by_variables, new_column, test=None):
    # Calculate mean values for 'grouped_variable' grouped by 'by_variables'
    means = train.groupby(by_variables)[grouped_variable].mean().reset_index()
    means.columns = by_variables + ['mean_' + grouped_variable]

    # Apply mean difference calculation function to train and test sets
    train = calculate_mean_difference_inner(train, grouped_variable, by_variables, new_column, means)
    
    if test:
        test = calculate_mean_difference_inner(test, grouped_variable, by_variables, new_column, means)

    return train, test



# RECLASSIFYING
# # Option 1: finished cycle [2nd cycle, 3rd cycle, high school, higher education] (ordinal)
# def get_ordinal_qualification(qualification):
#     if qualification == 'No school':
#         return 0    # No education
    
#     elif qualification in [f'{i}th grade' for i in range(3, 12)]:
#         return 1    # Basic education
    
#     elif '12' in qualification or qualification == 'Incomplete Bachelor\'s':
#         return 2    # Intermidiate education
    
#     else:
#         return 3    # Advanced education


# Option 2: year of education (numerical)
def get_years_of_education(qualification):
    years = re.findall(r'\d+', qualification)

    if qualification == 'No school':
        return 0    # No education
    
    elif years:
        return int(years[0])    # Total years
    
    elif qualification == 'Incomplete Bachelor\'s':
        return 13   # Considering 1 year in university
    
    elif qualification == 'Bachelor degree':
        return 15   # Bachelor's duration general rule is 3 years
    
    elif qualification == 'Post-Graduation':
        return 16   # Plus 1 year after Bachelor degree
    
    elif qualification == 'Master degree':
        return 17   # Plus 2 years after Bachelor degree
    
    elif qualification == 'PhD':
        return 21   # Plus 4 years after Master degree


##### FEATURE SELECTION

# CHI-SQUARE
def TestIndependence(X, y, var, alpha=0.05):
    '''
    Test the independence of a categorical variable with respect to the target variable.

    Parameters:
    -----------
    X : pandas.Series or array-like
        The independent categorical variable.

    y : pandas.Series or array-like
        The target variable.

    var : str
        The name of the variable being tested for importance.

    alpha : float, optional (default=0.05)
        The significance level for the test.

    Returns:
    --------
    None

    Notes:
    ------
    - Performs a chi-squared test of independence between X and y.
    - Compares the p-value with the significance level (alpha).
    - Prints whether the variable is important for prediction or not based on p-value.
    '''
    # Create a contingency table of observed frequencies
    dfObserved = pd.crosstab(y, X)
    
    # Perform chi-squared test and retrieve test statistics
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    
    # Create a DataFrame of expected frequencies
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index=dfObserved.index)
    
    # Determine the importance of the variable based on the p-value
    if p < alpha:
        important = True

    else:
        important = False
    

    return important


# RFE
def rfe(X, y):
    # Define a range of number of features to select
    nof_list = np.arange(1, len(X.columns))

    # Iterate through the range
    for n in range(len(nof_list)):
        # Initialize logistic regression model
        model = LogisticRegression(random_state=16)
        
        # Initialize Recursive Feature Elimination (RFE)
        rfe = RFE(model, n_features_to_select=nof_list[n])
        
        # Fit RFE on the data
        rfe.fit(X, y)

    # Return the selected features
    return rfe.support_



# SEQUENTIAL
def sequential_feature_selection(X, y):
    # Define the pipeline with SequentialFeatureSelector
    pipeline = Pipeline([
        ('sfs', SequentialFeatureSelector(LogisticRegression(random_state=16), 
                                          direction='forward', 
                                          scoring='f1_weighted', 
                                          n_jobs=-1))
    ])

    # Fit the pipeline on the data
    pipeline.fit(X, y)

    # Return the selected features
    return pipeline['sfs'].support_



# TREE-BASED
def tree_based_method(X, y, threshold='median'):
    '''
    Perform feature selection using the Extra Trees Classifier method.

    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        The feature matrix.

    y : pandas.Series or array-like
        The target variable.

    threshold : str or float, optional (default='median')
        The feature importance threshold to select features.

    Returns:
    --------
    None

    Notes:
    ------
    - Uses Extra Trees Classifier for feature selection based on feature importances.
    - Prints the names of selected features based on the specified threshold.
    - Does not modify the original data.
    - Reference: https://scikit-learn.org/stable/modules/feature_selection.html
    '''
    # Initialize Extra Trees Classifier
    rf_model = ExtraTreesClassifier(n_estimators=100, random_state=16)

    # Fit the model to your data
    rf_model.fit(X, y)

    # Create a feature selector based on feature importances
    feature_selector = SelectFromModel(rf_model, prefit=True, threshold=threshold)

    # Return the selected feature indices
    return feature_selector.get_support()


# LASSO
def lasso_method(X, y):
    '''
    Perform feature selection using Lasso regression and visualize feature importance.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    Returns:
    --------
    None

    Notes:
    ------
    - Fits a LassoCV model to perform feature selection.
    - Plots and visualizes feature importance based on Lasso coefficients.
    '''
    # Fit LassoCV model
    reg = LassoCV()
    reg.fit(X, y)
    
    # Extract feature coefficients and index them with column names
    coef = pd.Series(reg.coef_, index=X.columns)
    coef_selected = coef[abs(coef) > 0.015]
        
    selected_features = []
    # Get the selected features by Lasso
    for col in X.columns:
        if col in coef_selected.keys():
            selected_features.append(True)
        else:
            selected_features.append(False)
            
    return selected_features



# GENERAL
def feature_selection_cv(X, y, scaler, imputer, threshold=2, min_freq=8, show=False, export_name=None):
    # Select boolean columns
    columns_bool = X.select_dtypes(include=['bool']).columns

    # Initialize stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=16)

    # Dictionary to store results
    dict_results = {}

    # Iterate over cross-validation folds
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Split data into train and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Data transformation
        X_train, X_test = data_transform(scaler, X_train, X_test)
        X_train, X_test = data_transform(imputer, X_train, X_test)

        ##### CATEGORICAL
        # Test Independence
        bool_variables_to_use = []
        for col in columns_bool:
            important = TestIndependence(X_train[col], y_train, col)
            if important:
                bool_variables_to_use.append(col)

        ##### NUMERICAL
        # RFE
        rfe_features = rfe(X_train, y_train)

        # Sequential
        sequential_features = sequential_feature_selection(X_train, y_train)

        # Tree-based
        tree_features = tree_based_method(X_train, y_train)

        # LASSO
        lasso_features = lasso_method(X_train, pd.Categorical(y_train).codes)

        ##### DECISIONS
        features = pd.DataFrame({
            'RFE': rfe_features,
            'Sequential': sequential_features,
            'Tree-based': tree_features,
            'Lasso': lasso_features},
            index=X_train.columns)

        features_sum = features.sum(axis=1)
        features_bool = (features_sum > threshold)

        for var, count in features_sum.items():
            if var in bool_variables_to_use and count == threshold:
                features_bool[var] = True

        # Display if specified
        if show:
            print(i)
            features['DECISION'] = features_bool
            features.loc['TOTAL'] = features.sum(axis=0)
            # display(features)

        # Store results
        dict_results[i] = features_bool

    # Convert results to DataFrame
    results = pd.DataFrame(dict_results, index=X.columns).T

    # Calculate feature frequency
    features_freq = results.sum()

    # Display if specified
    if show:
        print('\nFEATURES FREQUENCY')
        print(features_freq)

    # Select features with frequency greater than minimum frequency
    features_to_use = features_freq[features_freq > min_freq - 1].index

    # Update X with selected features
    X = X[features_to_use]

    # Export if specified
    if export_name:
        X.to_csv(f'temp\\feature_selection\\{export_name}.csv')

    return X[features_to_use]
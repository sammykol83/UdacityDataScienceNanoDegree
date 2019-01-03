# This is a utility functions file to be used by the python notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

def MyScatterPlot(x, y, x_string, y_string, ax):
    '''
    INPUT:
    x - some array of numerical values
    y - some array of numerical values
    x_string - a X-axis label 
    y_string - a Y-axis label
    ax - the axis limits in the format [x_min, x_max, y_min, y_max]. Can also be empty.

    OUTPUT:
    None.

    The function generates a scatter plot of the 'x' vs. 'y'.
    '''
    # TODO: Move to utility file + documentation
    LARGE_SIZE = 16
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, color='b', marker='.')
    plt.xlabel(x_string)
    plt.ylabel(y_string)
    plt.grid(which='minor', axis='both')
    plt.rc('font', size=LARGE_SIZE)          # controls default text sizes
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.tick_params(labelsize=LARGE_SIZE)

    if ~(ax == []):
        plt.axis(ax)


def CalcSolarDensityVsMeanOfColumn( df, col_name, col_grid ):
    '''
    INPUT:
    df - A pandas dataframe from the DeepSolar dataframe
    col_name - some numerical column in the dataframe other than 'sol_density'
    col_grid - A grid of values that the column will be quantized into. e.g. the grid can be:
               [0,1,2,3,...,100] and all values of the column will be rounded to the nearest grid point.

    OUTPUT:
    An array of the AVERAGED solar systems installations per thousand households across the grid point. 

    The function help plotting the 'sol_density' column vs. Other columns. How? Since we have a lot of data, a straight
    forward scatter plot looks bad. Instead, we take any data column and reduce the number of points within it by "binning" of it's values
    to a smaller grid. (Just like in a histogram). Then, per each bin, we calculate an average of all 'sol_density' values we have.
    '''
    
    # Init empty result list
    solar_systems_per_thousand_households = []
    
    # Quantize the column value according to the supplied grid, get back the indices
    indices = np.digitize( df[col_name], np.sort(col_grid) )
    
    # Go over grid values
    for val in range(1,len(col_grid)+1):
        
        # Get susbset of table 
        tmp = df.loc[ indices == val]
        
        # Append the mean of solar systems density
        solar_systems_per_thousand_households.append(np.mean(tmp['sol_density']))    

    return solar_systems_per_thousand_households

def RandomForestRegressorWithKFold(X, y, n_splits, random_state):
    '''
    INPUT:
    X - A pandas dataframe from the DeepSolar dataframe. 
    y - A different feature not included in 'X' that we want to predict. 
    n_splits - Number of splits for KFold that is performed within the function. 
    random_state - the random state used for KFold that is performed within the function.

    OUTPUT: 
    (NOTICE - The returned values are for the highest R2 score)
    model - The model after training
    y_test - The test set labels (actually values)
    y_test_preds - The predicted labels( actually values). 

    The function uses a simple RandomForestRegressor combined with a KFold splitting of the data for performing regression. 
    on some target variable. The function then prints the R2 metric for each split and the average for all splits. 
    '''
        
    # Init
    r2_score_list = []
    model_tmp_list = []
    y_test_preds_list = []
    y_test_list = []
        
    # Split data into n_splits, where n_splits-1 are for training and 1 split for testing. 
    kf = KFold(n_splits = n_splits, random_state = random_state, shuffle=True)
    
    # Go over the training / testing indices from the split
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Generate and fit the RandomForestRegressor() model
        model = RandomForestRegressor() # Instantiate
        model.fit(X_train, y_train)     #Fit

        #Predict and score the model
        y_test_preds = model.predict(X_test) 
        r2_score_list.append( r2_score(y_test, y_test_preds) )
        print( "The R-squared score for the model was %2.5f on %d values." % (r2_score(y_test, y_test_preds), len(y_test)))
        
        # Save values for returning later
        model_tmp_list.append(model)
        y_test_list.append(y_test)
        y_test_preds_list.append(y_test_preds)

    mean_r2_score_list = np.mean( r2_score_list )    
    print('The average R-squared score is: %2.5f' % mean_r2_score_list )
    
    model = model_tmp_list[np.argmax( r2_score_list )]
    y_test_preds = y_test_preds_list[np.argmax( r2_score_list )]
    y_test = y_test_list[np.argmax( r2_score_list )]
    
    return model, y_test_preds, y_test
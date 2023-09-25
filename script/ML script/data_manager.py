# data_manager.py
# Load and save DF from / to CSV

# 2023-03-16: added drop_duplicates parameter in data_load

import pandas as pd

import os

def data_load(file_name, dir_name, cols_select, drop_duplicates = True):
    """ Given a csv file_name, load it into a pandas Dataframe; key_index is the custom index of the DF """    
    path_input = dir_name + os.sep + file_name
    print()
    print("Reading csv input:", path_input)
    print("Columns filter (input):", str(cols_select))
    if len(cols_select) > 0:
        if 'cig' in cols_select:
            input_df = pd.read_csv(path_input, delimiter = ";", usecols = cols_select, dtype = {'cig':object, 'cod_modalita_realizzazione': object, 'cod_tipo_scelta_contraente':object}, low_memory = False)
        else:
            input_df = pd.read_csv(path_input, delimiter = ";", usecols = cols_select, low_memory = False)
    else:
        input_df = pd.read_csv(path_input, delimiter = ";", dtype = {'cig':object, 'cod_modalita_realizzazione': object, 'cod_tipo_scelta_contraente': object}, low_memory = False)
    print("...done!")
    
    # Search for duplicate rows, count and delete them
    if drop_duplicates == True:
        if input_df.duplicated().sum() > 0:
            print("Rows duplicated (to be deleted): ", input_df.duplicated().sum())
            input_df = input_df.drop_duplicates()

    print()
    return input_df


def data_show(input_df, order_by = ""):
    """ Given a pandas Dataframe, show the data """
    print("-"*30)
    print()
    print("Dataframe description")
    print()
    print("Dataframe length (rows):", len(input_df)) # Show number of rows
    print()
    print("Dataframe shape (rows, cols):", input_df.shape) # Show rows,cols number
    print()
    print("Dataframe columns:", input_df.columns) # Show columns (features)
    print()
    print("Column types:")
    print()
    print(input_df.dtypes) # Column types
    print()
    print(input_df.dtypes.value_counts())
    print()
    if order_by != "" and order_by in input_df: # check if order_by is not empty and if exist in DF column 
        input_df.sort_values(by = order_by, ascending = False, inplace = True)
    print(input_df.head()) # Show first 10 rows
    print("...")
    print(input_df.tail()) # Show last 10 rows
    print()
    print("-"*30)
    print()


def data_save(input_df, file_name, dir_name, index_keep):
    """ Given a DF, save it to CSV file (index_keep if True save the index too) """
    path_output = dir_name + os.sep + file_name
    input_df.to_csv(path_output, sep = ";", index = index_keep, header = True)


def data_appeal(file_name, dir_name):
    """Get the csv with appeals to add it to main dataframe """
    path_input = dir_name + os.sep + file_name
    print()
    print("Reading csv input '",path_input,"'")
    print()
    cig_df = pd.read_csv(path_input, delimiter = ";", usecols=[0], names=['cig'], dtype = {'cig':object}) # read the first column of the csv and add the header 'cig' of type string (object)
    print("...done!")
    print()
    cig_df = cig_df.drop_duplicates()
    cig_list = cig_df['cig'].tolist()
    return cig_df, cig_list
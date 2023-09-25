
# 07_ml_data_model.py

# 2023-03-10: added ANN (Artificial Neural Network) -> commented
# 2023-03-18: remove method_implementation from num_cols, added OneHotEnc (modality_*)

# GLOBAL imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Visualization
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns # visualizing confusion_matrix

# Metrics library from scikit-learn
# https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score

# Preprocessing 
# https://scikit-learn.org/stable/modules/preprocessing.html#
from sklearn.preprocessing import StandardScaler

# ML Algorithms
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost import XGBClassifier
# pip install -U xgboost

# Hyperparameter Tuning (HT)
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

# save the model
import joblib

# ANN
# pip install tensorflow-macos
# pip install keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
# for modeling
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping

import shap

# prettify
from pprint import pprint

# LOCAL imports
import config_reader
import timing
import data_manager
import log_manager

### GLOBALS ###

yaml_config = config_reader.read_yaml()
data_dir = yaml_config['DATA_DIR']
charts_dir = yaml_config['CHARTS_DIR']
hp_tuning = yaml_config['HP_TUNING'] 

dim_sample = -1  # <- INPUT number of rows to consider for MLA (dim_sample*2 -> dim_sample with label 1, dim_sample with label 0); -1 for all
shap_estimate = 0 # <- INPUT (0 no, 1 yes)

tender_type_list = ["SERVIZI", "LAVORI", "FORNITURE"]
# tender_type_list = ["SERVIZI"]

# tender_type = "SERVIZI" # <-- INPUT (SERVIZI, LAVORI, FORNITURE)
# test_size_perc = 0.2 # INPUT: - > 0.1, 0.2, 0.3, test 0.2 -> 20% -> train 80%

log_file = "07_ml_data_model_#.txt" # <-- INPUT

test_size_perc_list = [0.1, 0.2, 0.3]
# test_size_perc_list = [0.1, 0.2]
# test_size_perc_list = [0.2]

# dictionary to translate procurement type (object) in english
dic_procurement_object = {"SERVIZI":"SERVICES", "LAVORI":"PUBLIC WORKS", "FORNITURE":"GOOD/SUPPLIES"}

# dictionary XGB Hyper Params
dic_xgb_hp = {'SERVIZI':{'colsample_bytree': 0.5738327279350487, 'gamma': 4.285912652704157, 'max_depth': 6, 'min_child_weight': 8, 'reg_alpha': 40, 'reg_lambda': 0.9328068435894733}, 'LAVORI': {'colsample_bytree': 0.7960743157146718, 'gamma': 7.775764837497972, 'max_depth': 12, 'min_child_weight': 3, 'reg_alpha': 115, 'reg_lambda': 0.015451851019482099}, 'FORNITURE': {'colsample_bytree': 0.5105097441251072, 'gamma': 5.331991250752087, 'max_depth': 17, 'min_child_weight': 6, 'reg_alpha': 58, 'reg_lambda': 0.6420628246186448}}

# Dataframe to collect Models metrics
MLA_columns = ['MLA Name', 'PosCases', 'NegCases', 'FPR', 'FNR', 'TNR', 'TPR', 'PPV', 'Accuracy', 'F1Score'] # PosCases with label 1 (appeal present)
MLA_compare = pd.DataFrame(columns = MLA_columns)

MLA_roc_columns = ['MLA Name', 'PosCases', 'NegCases',  'TPR', 'FPR', 'AUC']
MLA_roc_values = pd.DataFrame(columns = MLA_roc_columns) # to collect all the ROC/AUC 

MLA_index = 0

std = StandardScaler()

conf_matrix_all = {}

a = []

### FUNCTIONS ###

def sample_random_rows(input_df, dim):
    """ Get a subset of cases with appeal = 0 and appeal = 1 """
    df_1 = input_df[input_df.appeal == 1] # dataframe with label = 1
    df_0 = input_df[input_df.appeal == 0] # dataframe with label = 0
    df_0_rows_index = np.random.choice(df_0.index.values, dim) # get dim random rows with label = 0
    df_0_rows = df_0.loc[df_0_rows_index] # generate the dataframe 
    df_1_rows_index = np.random.choice(df_1.index.values, dim) # get dim random rows with label = 0
    df_1_rows = df_1.loc[df_1_rows_index] # generate the dataframe 
    frames = [df_0_rows, df_1_rows]
    result_df = pd.concat(frames) # concatenate two dataframes
    print("Outcome = 1 in DF:", len(result_df[result_df['appeal']==1]))
    print("Outcome = 0 in DF:", len(result_df[result_df['appeal']==0]))
    print()
    return result_df

def roc_curve_plot(tender_type, MLA_roc_values):
    """ Plot MLA_roc_values dataframe"""
    # figure(figsize=(16, 12), dpi=80)
    global dic_procurement_object
    global dim_sample
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    # sample: plt.plot(fpr, tpr, marker='.', label='Random Forest (AUC = %0.3f)' % auc)
    MLA_roc_values.sort_values(by = 'AUC', ascending = False, inplace = True)
    # for (_, col1, col2, col3, col4) in MLA_roc_values.itertuples():
    for row in MLA_roc_values.itertuples():
    # col0 --> row index
    # col1 --> 'MLA Name'
    # col2 --> 'PosCases'
    # col3 --> 'NegCases'
    # col4 --> 'TPR'
    # col5 --> 'FPR'
    # col6 --> 'AUC'
        name = row[1]
        auc = row[6]
        # auc = col4
        label_name = name + ' (AUC = %0.3f)' % auc
        tpr = row[4]
        # tpr = col2
        fpr = row[5]
        plt.plot(fpr, tpr, marker='.', label = label_name)

    plt.plot([0,1], [0,1], 'k--') # dashed diagonal
    # plt_title = "Metric: ROC " + "\n" + "Procurement object: " + dic_procurement_object[tender_type] + " (cases: " + str(dim_sample*2) + ")"
    plt_title = "Metric: ROC " + "\n" + "Procurement object: " + dic_procurement_object[tender_type] + "\n" + " Test size: " + str((test_size_perc*100))
    plt.title(plt_title, fontweight = 'bold')
    plt.grid(visible=True)
    plt.xlabel('False Positive Rate (FPR)', fontweight = 'bold')
    plt.ylabel('True Positive Rate (TPR or recall)', fontweight = 'bold')
    plt.legend()
    print("Drawing...")
    plot_file_name = "07_ML_" + tender_type + "_ROC-AUC_all_"+str((test_size_perc*100))+".png"
    path_plot = charts_dir + os.sep + plot_file_name
    plt.savefig(path_plot)
    # plt.show()
    plt.clf()
    print("...done!")
    print()
    

def data_plot(tender_type, key):
    """ Give a Dataframe of values, plot the horizontal bar chart """
    global MLA_compare
    global dic_procurement_object
    global dim_sample
    global test_size_perc
    # print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    MLA_compare.sort_values(by = [key], ascending = False, inplace = True)

    #barplot using Seaborn: https://seaborn.pydata.org/generated/seaborn.barplot.html
    ax = sns.barplot(x = key, y = 'MLA Name', data = MLA_compare)
    ax.bar_label(ax.containers[0])

    #prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
    # plt_title = "Metric: " + key + "\n" + " Procurement object: " + dic_procurement_object[tender_type] + " (cases: " + str(dim_sample*2) + ")"
    plt_title = "Metric: " + key + "\n" + " Procurement object: " + dic_procurement_object[tender_type] + "\n" + " Test size: " + str((test_size_perc*100))
    plt.title(plt_title, fontweight = 'bold')
    plt.xlabel('Score (%)', fontweight = 'bold')
    plt.ylabel('Algorithm', fontweight = 'bold')
    plt.tight_layout() # avoid span borders
    print("Drawing...")
    plot_file_name = "07_ML_" + tender_type + "_ACCURACY_all_" + str((test_size_perc*100)) + ".png"
    path_plot = charts_dir + os.sep + plot_file_name
    plt.savefig(path_plot)
    # plt.show()
    plt.clf()
    print("...done!")
    print()

def shap_plot(model, train_X, tender_type):
    """Explain the model's predictions using SHAP"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_X)
    shap.summary_plot(shap_values, train_X, plot_type="bar")
    # Features with large absolute Shapley values are important. Since we want the global importance, we average the absolute Shapley values per feature across the data.
    print("Drawing SHAP...")
    plot_file_name = "07_ML_" + tender_type + "_SHAP.png"
    path_plot = charts_dir + os.sep + plot_file_name
    plt.savefig(path_plot)
    plt.show()
    plt.clf()
    print("...done!")
    print()

def event_prediction(model_name, algo, training_x, testing_x, training_y, testing_y, plot, tender_type, test_size_perc):
    """ Function for predictions """
    global a
    global dic_procurement_object
    global MLA_index 
    algo.fit(training_x,training_y)                           # Fit the training data set to the algorithm passed.
    predictions = algo.predict(testing_x)                     # Get all predictions (y_pred)
    probabilities = algo.predict_proba(testing_x)             # Get probablities of predictions

    conf_matrix = confusion_matrix(testing_y, predictions)    # Get confusion matrix using the predictions
    tn, fp, fn, tp = conf_matrix.ravel()
    
    conf_matrix_all[model_name] = conf_matrix                  # Save confusion matrix values to a dictionary
    a = conf_matrix

    false_positive_rate = fp / (fp + tn) # FPR
    false_negative_rate = fn / (tp + fn) # FNR
    true_negative_rate = tn / (tn + fp)  # TNR
    recall = recall_score(testing_y, predictions) # TPR: tp / (tp + fn)
    precision = precision_score(testing_y, predictions) # PPV: tp / (tp + fp)
    accuracy = accuracy_score(testing_y, predictions) # ACC: (tp + tn) / (tp + fp + fn + tn)
    f1 = f1_score(testing_y, predictions) # the harmonic mean between precision and recall    
    
    print("Classification report (",model_name,") for tender type",dic_procurement_object[tender_type], "with test size:", test_size_perc)  # Print the classification report
    print(classification_report(testing_y, predictions))

    # model_roc_auc = roc_auc_score(testing_y, predictions)           # Get the Area under the curve number
    model_roc_auc = roc_auc_score(testing_y, probabilities[:,1])           # Get the Area under the curve number
    fpr,tpr,thresholds = roc_curve(testing_y, probabilities[:,1])   # Get False postive rate and true positive rate

    print("Area under the curve:", model_roc_auc)
    print("Accuracy score:", accuracy_score(testing_y, predictions))
    print()
    
    # save metrics
    MLA_compare.loc[MLA_index, 'MLA Name'] = model_name
    #  MLA_compare.loc[MLA_index, 'TestSize'] = test_size_perc*100
    MLA_compare.loc[MLA_index, 'PosCases'] = dim_sample
    MLA_compare.loc[MLA_index, 'NegCases'] = dim_sample    
    MLA_compare.loc[MLA_index, 'FPR'] = false_positive_rate
    MLA_compare.loc[MLA_index, 'FNR'] = false_negative_rate
    MLA_compare.loc[MLA_index, 'TNR'] = true_negative_rate
    MLA_compare.loc[MLA_index, 'TPR'] = recall
    MLA_compare.loc[MLA_index, 'PPV'] = precision
    MLA_compare.loc[MLA_index, 'Accuracy'] = accuracy
    MLA_compare.loc[MLA_index, 'F1Score'] = f1

    # save metrics for ROC / AUC plot
    MLA_roc_values.loc[MLA_index, 'MLA Name'] = model_name
    # MLA_roc_values.loc[MLA_index, 'TestSize'] = test_size_perc*100
    MLA_roc_values.loc[MLA_index, 'PosCases'] = dim_sample
    MLA_roc_values.loc[MLA_index, 'NegCases'] = dim_sample
    MLA_roc_values.loc[MLA_index, 'TPR'] = tpr
    MLA_roc_values.loc[MLA_index, 'FPR'] = fpr
    MLA_roc_values.loc[MLA_index, 'AUC'] = model_roc_auc
    MLA_index+=1

    # save the RF model 
    if model_name=="RandomForestClassifier":
        file_model = "RF_uncompressed_" + tender_type + "_" + str(test_size_perc) + ".joblib"
        joblib.dump(algo, file_model, compress=0) # save the model
        print(f"Uncompressed Random Forest: {np.round(os.path.getsize(file_model) / 1024 / 1024, 2) } MB")

    if plot:

        fig, axes = plt.subplots(1,2, figsize=(25, 5))
        conf_matrix = np.flip(conf_matrix)
        
        conf_2 = conf_matrix.astype(str)
        labels = np.array([['\nTP','\nFN'],['\nFP','\nTN']])
        labels = np.core.defchararray.add(conf_2, labels)
        sns.heatmap(conf_matrix, fmt='', annot = labels, ax=axes[0], cmap="YlGnBu", xticklabels=[1, 0], yticklabels=[1, 0]) # Plot the confusion matrix
        title_text = "Model name: " + model_name + "\n" + "Confusion matrix (" +dic_procurement_object[tender_type]+")"
        axes[0].set(xlabel='Predicted', ylabel='Actual', title = title_text)

        title_text = "Model name: " + model_name + "\n" + "ROC curve (" +dic_procurement_object[tender_type]+")"
        plt.title(title_text)
        sns.lineplot(x = fpr, y = tpr, ax = axes[1])                               # Plot the ROC curve
        plt.plot([0, 1], [0, 1],'--')                                              # Plot the diagonal line
        axes[1].set_xlim([0, 1])                                                   # Set x-axis limit to 0 and 1
        axes[1].set_ylim([0, 1])                                                   # Set y-axis limit to 0 and 1
        axes[1].set(xlabel = 'False Positive Rate (FPR)', ylabel = 'True Positive Rate (TPR)')
        plt.legend()
        print("Drawing...")
        plt.tight_layout() # avoid span borders
        plot_file_name = "07_ML_" + tender_type + "_CM-ROCAUC_"+model_name + "_" + str((test_size_perc*100)) + ".png"
        path_plot = charts_dir + os.sep + plot_file_name
        plt.savefig(path_plot)
        # plt.show()
        plt.clf()
        print("...done!")
        print()

        


def event_prediction_ann(X, y, tender_type):
    global MLA_index 
    XN = np.array(X) # for ANN
    # build a model
    model = Sequential()
    model.add(Flatten(input_shape=(XN.shape[1],))) # input layer 
    # model.add(Dense(16, input_shape=(XN.shape[1],), activation='relu')) # alternative to Flatten input layer (with shape)
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu')) # added
    model.add(Dense(16, activation='relu')) # added
    model.add(Dense(1, activation='sigmoid'))
    model.summary() 
    # compile the model
    model.compile(optimizer='Adam', 
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # early stopping callback
    # This callback will stop the training when there is no improvement in  
    # the validation loss for 10 consecutive epochs.  
    es = EarlyStopping(monitor='val_accuracy', 
                                    mode='max', # don't minimize the accuracy!
                                    patience=10,
                                    restore_best_weights=True)

    # now we just update our model fit call
    history = model.fit(XN,
                        y,
                        callbacks=[es],
                        epochs=80, # you can set this to a big number!
                        batch_size=10,
                        validation_split=0.2,
                        shuffle=True,
                        verbose=1)

    # ANN metrics (Evaluate the Model)
    # Learning curves (Loss)
    # Learning curves (Accuracy)
    # Confusion matrix

    history_dict = history.history

    # Learning curve(Loss)
    # let's see the training and validation loss by epoch

    # loss
    loss_values = history_dict['loss'] # you can change this
    val_loss_values = history_dict['val_loss'] # you can also change this

    # range of X (no. of epochs)
    epochs = range(1, len(loss_values) + 1) 

    # plot
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.clf()

    # Learning curve (accuracy)
    # let's see the training and validation accuracy by epoch

    # accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # range of X (no. of epochs)
    epochs = range(1, len(acc) + 1)

    # plot
    # "bo" is for "blue dot"
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    # orange is for "orange"
    plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()
    plt.clf()

    # this is the max value - should correspond to
    # the HIGHEST train accuracy
    np.max(val_acc)

    # confusion matrix and classification report

    # see how these are numbers between 0 and 1? 
    model.predict(XN) # prob of successes (appeal)
    np.round(model.predict(XN),0) # 1 and 0 (appeal or not)

    # so we need to round to a whole number (0 or 1), or the confusion matrix won't work!
    preds = np.round(model.predict(XN),0)

    # confusion matrix
    print("ANN confusion matrix:")
    conf_matrix = confusion_matrix(y, preds)
    tn, fp, fn, tp = conf_matrix.ravel()
    print(confusion_matrix(y, preds)) # order matters! (actual, predicted)
    print()

    ## array([[...,  ...],   ([[TN, FP],
    ##       [..., ...]])     [FN, TP]])

    fpr, tpr, thresholds = metrics.roc_curve(y, preds, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    model_roc_auc = roc_auc_score(y, preds)

    print("ANN classification report")   
    print(classification_report(y, preds))
    print(roc_auc)

    false_positive_rate = fp / (fp + tn) # FPR
    false_negative_rate = fn / (tp + fn) # FNR
    true_negative_rate = tn / (tn + fp)  # TNR
    recall = recall_score(y, preds) # TPR: tp / (tp + fn)
    precision = precision_score(y, preds) # PPV: tp / (tp + fp)
    accuracy = accuracy_score(y, preds) # ACC: (tp + tn) / (tp + fp + fn + tn)
    f1 = f1_score(y, preds) # the harmonic mean between precision and recall

    # save metrics
    MLA_compare.loc[MLA_index, 'MLA Name'] = 'ANN'
    MLA_compare.loc[MLA_index, 'PosCases'] = dim_sample
    MLA_compare.loc[MLA_index, 'NegCases'] = dim_sample    
    MLA_compare.loc[MLA_index, 'FPR'] = false_positive_rate
    MLA_compare.loc[MLA_index, 'FNR'] = false_negative_rate
    MLA_compare.loc[MLA_index, 'TNR'] = true_negative_rate
    MLA_compare.loc[MLA_index, 'TPR'] = recall
    MLA_compare.loc[MLA_index, 'PPV'] = precision
    MLA_compare.loc[MLA_index, 'Accuracy'] = accuracy
    MLA_compare.loc[MLA_index, 'F1Score'] = f1 


    # save metrics for ROC / AUC plot
    MLA_roc_values.loc[MLA_index, 'MLA Name'] = 'ANN'
    # MLA_roc_values.loc[MLA_index, 'TestSize'] = test_size_perc*100
    MLA_roc_values.loc[MLA_index, 'PosCases'] = dim_sample
    MLA_roc_values.loc[MLA_index, 'NegCases'] = dim_sample
    MLA_roc_values.loc[MLA_index, 'TPR'] = tpr
    MLA_roc_values.loc[MLA_index, 'FPR'] = fpr
    MLA_roc_values.loc[MLA_index, 'AUC'] = model_roc_auc
    MLA_index+=1

    MLA_index+=1


### MAIN ###

print()
print("*** ML data models task 7 (applying algorithms) ***")
print()

timing.timing_start()

for tender_type in tender_type_list:

    file_name = "02_bando_cig_"+tender_type+"_label_balanced.csv" 

    input_df = data_manager.data_load(file_name, data_dir, [], False) # don't drop the duplicates

    for test_size_perc in test_size_perc_list:

        print("Tender type:", tender_type)

        print("Test size:", test_size_perc)

        log_file = log_file.replace("#", tender_type)

        # empty the resulst df
        MLA_compare.drop(MLA_compare.index,inplace=True)
        MLA_roc_values.drop(MLA_roc_values.index,inplace=True)

        with open(log_file, "w") as fp:
            fp.write("ML models on tender type: " + tender_type + os.linesep)
            fp.write(timing.timing_start_str() + os.linesep)
            fp.write("******" + os.linesep)

        ####### SELECTION #######
        # TO DO

        # columns to be removed
        # ALL
        # cols_remove = ["cig","framework_agreement","lots","importo_lotto","cod_tipo_scelta_contraente","cod_modalita_realizzazione","anno_pubblicazione","cod_cpv","numero_offerte_ammesse","importo_aggiudicazione","ribasso_aggiudicazione","delta_date","settori_ordinari","settori_speciali","ABRUZZO","BASILICATA","CALABRIA","CAMPANIA","CENTRALE","EMILIA-ROMAGNA","FRIULI-VENEZIA-GIULIA","LAZIO","LIGURIA","LOMBARDIA","MARCHE","MOLISE","NC","PA-BOLZANO","PA-TRENTO","PIEMONTE","PUGLIA","SARDEGNA","SICILIA","TOSCANA","UMBRIA","VDAOSTA","VENETO","appeal"] 
        
        # cols_remove = []
        cols_remove = ["cig", "year"] 

        with open(log_file, "a") as fp:
            fp.write("DF columns (features): " + str(input_df.columns) + os.linesep)
            fp.write("DF columns (features) removed: " + str(cols_remove) + os.linesep)
            fp.write("******" + os.linesep)

        print("Cols removed:", str(cols_remove))

        for col in cols_remove:
            if col in input_df.columns:
                input_df.drop(col, inplace = True, axis = 1)

        print("Data read from csv:")
        data_manager.data_show(input_df)
        print()

        # resize the DF for quick tests
        if dim_sample > 0:
            print("Data smaller:")
            input_df = sample_random_rows(input_df, dim_sample)
            data_manager.data_show(input_df)
            print()
        else:
            print("Original size sample:", len(input_df))

        col_names = input_df.columns[:-1]
        print("Columns (features):")
        print(col_names)
        target_col = ["appeal"] # label / target / outcome column
        print("Target col (outcome):")
        print(target_col)

        # numeric columns
        # all
        num_cols = ["lots", "procurement_amount", "cod_cpv", "bids_admitted", "bid_award", "bid_drop", "year", "delta_date"] 
        
        # remove required columns cols_remove
        for element in num_cols:
            if element in cols_remove:
                num_cols.remove(element)

        # remove missing columns (because missing in original df like some missing modality)
        for element in num_cols:
            if element not in input_df.columns[:-1]:
                num_cols.remove(element)

        # remove the columns from cols_remove
        # num_cols = filter(lambda i: i not in cols_remove, num_cols)
        num_cols = [i for i in num_cols if i not in cols_remove]
        print("Numeric columns:", len(num_cols))

        # binary (1 / 0) columns
        # all
        # modality

        bin_cols = ["framework_agreement", "ordinary", "specials", "ABRUZZO","BASILICATA","CALABRIA","CAMPANIA","CENTRALE","EMILIA-ROMAGNA","FRIULI-VENEZIA-GIULIA","LAZIO","LIGURIA","LOMBARDIA","MARCHE","MOLISE","NC","PA-BOLZANO","PA-TRENTO","PIEMONTE","PUGLIA","SARDEGNA","SICILIA","TOSCANA","UMBRIA","VDAOSTA","VENETO", "consortium", "individual", "subcontract",'modality_1', 'modality_2', 'modality_3', 'modality_4', 'modality_5', 'modality_6', 'modality_7', 'modality_8', 'modality_9', 'modality_10', 'modality_11', 'modality_12', 'modality_13', 'modality_14', 'modality_15', 'modality_16', 'modality_17', 'modality_18', 'modality_19', 'eo_selection_1', 'eo_selection_2', 'eo_selection_3', 'eo_selection_4', 'eo_selection_5', 'eo_selection_6', 'eo_selection_7', 'eo_selection_8', 'eo_selection_12', 'eo_selection_14', 'eo_selection_15', 'eo_selection_16', 'eo_selection_22', 'eo_selection_23', 'eo_selection_24', 'eo_selection_25', 'eo_selection_26', 'eo_selection_27', 'eo_selection_28', 'eo_selection_29', 'eo_selection_30', 'eo_selection_32', 'eo_selection_33', 'eo_selection_34', 'eo_selection_35', 'eo_selection_36', 'eo_selection_37', 'eo_selection_38', 'eo_selection_40', 'eo_selection_114', 'eo_selection_122']
        
        # remove required columns cols_remove
        for element in bin_cols:
            if element in cols_remove:
                bin_cols.remove(element)

        # remove missing columns (because missing in original df like some missing modality)
        for element in bin_cols:
            if element not in input_df.columns[:-1]:
                bin_cols.remove(element)

        # remove the columns from cols_remove
        # bin_cols = filter(lambda i: i not in cols_remove, bin_cols)
        bin_cols = [i for i in bin_cols if i not in cols_remove]
        print("Binary columns:", len(bin_cols))

        df_summary = input_df.describe()
        print(input_df.describe())
        file_name = "07_bando_cig_"+tender_type+"_MLA-input_describe.txt" 
        path_out = data_dir + os.sep + file_name
        df_summary.to_csv(file_name, sep = ";")

        # Check if any of the columns have null values (not necessary)
        print("Data read check:")
        print(input_df.isnull().sum())
        print()

        # scale the data
        scaled = std.fit_transform(input_df[num_cols])     # Standardize the numeric columns to get them on the same scale
        scaled = pd.DataFrame(scaled, index = input_df.index, columns=num_cols) # index = input_df.index keep the same indexing of input_df

        """
        print("Data scaled check:")
        data_show(scaled)
        print()
        print(scaled.isnull().sum())
        print()
        """

        df_train = pd.concat([scaled, input_df[bin_cols + target_col]], axis=1) # concat the scaled df with the remaining columns (binary and outcome)

        """
        print("Data for training ML check:")
        data_show(df_train)
        print()
        print(df_train.isnull().sum())
        print()
        """

        # heatmap
        """
        plt.figure(figsize=(15,10))
        ax = plt.axes()
        sns.heatmap(df_train.corr(), annot=True, fmt='.2g', ax = ax)
        text_title = "Heatmap for procurement type: " + tender_type
        ax.set_title(text_title)
        print("Drawing...")
        # plot_file_name = "02_bando_cig_" + tender_type + "_HEATMAP.png"
        # path_plot = charts_dir + os.sep + plot_file_name
        # plt.savefig(path_plot)
        # plt.show()
        """

        X = df_train[col_names]      # Contains the independent columns (matrix input for ML)
        XN = np.array(X)             # for ANN
        y = df_train[target_col]     # Target (outcome) column

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size_perc, random_state=42) # generate the dataframe for train (80% of rows) and test (20% of rows)
        train_y = train_y[target_col[0]]
        test_y = test_y[target_col[0]]

        train_X.to_csv("train_X"+str(test_size_perc)+".csv", index = False, header = True)
        test_X.to_csv("test_X"+str(test_size_perc)+".csv", index = False, header = True)
        train_y.to_csv("train_y"+str(test_size_perc)+".csv", index = False, header = True)
        test_y.to_csv("test_y"+str(test_size_perc)+".csv", index = False, header = True)

        lr  = LogisticRegression()
        event_prediction(lr.__class__.__name__ , lr, train_X, test_X, train_y, test_y, True, tender_type, test_size_perc) # True -> plot the metrics

        knn = KNeighborsClassifier()
        event_prediction(knn.__class__.__name__ , knn, train_X, test_X, train_y, test_y, True, tender_type, test_size_perc)

        svc = SVC(probability=True) # probability True to get predict_proba
        event_prediction(svc.__class__.__name__ , svc, train_X, test_X, train_y, test_y, True, tender_type, test_size_perc)

        dtc = DecisionTreeClassifier()
        event_prediction(dtc.__class__.__name__, dtc, train_X, test_X, train_y, test_y, True, tender_type, test_size_perc)

        rfc = RandomForestClassifier()
        event_prediction(rfc.__class__.__name__, rfc, train_X, test_X, train_y, test_y, True, tender_type, test_size_perc)

        xgb_param_drid = dic_xgb_hp[tender_type]
        xgc = XGBClassifier(**xgb_param_drid)
        # xgc = XGBClassifier()
        event_prediction(xgc.__class__.__name__, xgc, train_X, test_X, train_y, test_y, True, tender_type, test_size_perc)

        # event_prediction_ann(X, y, tender_type) # ANN not ready

        """ 
        # Random Hyperparameter Grid
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
        # Number of features to consider at every split
        max_features = ['sqrt', 'log2'] # 'auto' should be removed because deprecated
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap
                    }
        print("Random Grid:")
        pprint(random_grid)
        print()

        rfc = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(train_X,train_y)
        print("Best params for HT:")
        print(rf_random.best_params_)
        print()

        log_manager.log_writer(log_file, "a", "RF best params:" + str(rf_random.best_params_))

        # Results rf_random.best_params_: 

        """

        # best model Hyper Tuning
        if hp_tuning == 1:

            print("Random Forest HP tuning ...")

            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

            param_grid = {
                #'bootstrap': [True],
                'bootstrap': [False],
                'min_samples_leaf': [1, 2, 3],
                'min_samples_split': [2, 4, 6],
                'max_depth': [50, 100, 200], # None removed
                'n_estimators': [400, 800, 1000], 
                'n_jobs': [-1], # added: the number of jobs to run in parallel (-1 all processors)
                # 'oob_score': [True, False], # added
                'max_features': ['sqrt', 'log2', None] # 'auto' removed because deprecated
            }

            # instantiate the grid search model
            grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

            # fit the grid search to the data
            grid_search.fit(train_X, train_y) # grid_search is the new model

            model_name = grid_search.__class__.__name__ + " Hyper Tuning" # get the class name of the ML + HT

            print("RF best params:", grid_search.best_params_)
            print()

            with open(log_file, "a") as fp:
                fp.write("RF best params:" + str(grid_search.best_params_))

            event_prediction(model_name, grid_search, train_X, test_X, train_y, test_y, True, tender_type, test_size_perc)

            file_model = "RF_HT_uncompressed_" + tender_type + "_" + str(test_size_perc) + ".joblib"
            joblib.dump(grid_search, file_model, compress=0) # save the model
            print(f"Uncompressed Random Forest HT: {np.round(os.path.getsize(file_model) / 1024 / 1024, 2) } MB")

            print()


        MLA_compare.sort_values(by = 'Accuracy', ascending = False, inplace = True)
        path = data_dir + os.sep + "mla_compare_"+tender_type + "_" + str(test_size_perc*100) + ".csv"
        MLA_compare.to_csv(path, sep = ";", index = False)

        MLA_roc_values.sort_values(by = 'AUC', ascending = False, inplace = True)
        path = data_dir + os.sep + "mla_roc_" + tender_type + "_" + str(test_size_perc*100) + ".csv"
        MLA_roc_values.to_csv(path, sep = ";", index = False)

        print("-"*30)
        print()

        print("Metrics output:")
        print()
        data_manager.data_show(MLA_compare)
        print()

        print("Metrics ROC/AUC:")
        print()
        data_manager.data_show(MLA_roc_values)
        print()

        print("-"*30)
        print()

        # plot ROC
        roc_curve_plot(tender_type, MLA_roc_values)

        # plot accuracy
        data_plot(tender_type, 'Accuracy')

        # SHAP 

        if shap_estimate == 1:
            print("Explaining the model with SHAP...")
            shap_plot(rfc, train_X, tender_type) # slow

        print()

timing.timing_end()

with open(log_file, "a") as fp:
    fp.write(timing.timing_end_str() + os.linesep)
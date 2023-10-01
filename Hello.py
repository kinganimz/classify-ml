# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)
import streamlit as st
import pandas as pd
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts
from kennard_stone import train_test_split as ks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from keras.layers import Dense
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from io import BytesIO
from reportlab.platypus import Image
from sklearn.tree import plot_tree

## FUNCTIONS

# Function to create dataset
@st.cache_data
def create_dataset(df, X, y=None):
    X_df = pd.DataFrame(df[X])
    y_df = pd.DataFrame(df[y])
    return X_df.join(y_df)

# Function to separate categorical variables
@st.cache_data
def separate_categorical(X_df, cat_variables):
    return pd.get_dummies(X_df, columns=cat_variables)

# Function to split data
@st.cache_data
def split_data(X, y, method, split):
    if method == "Scikit-learn (random)":
        return tts(X, y, test_size=split, random_state=42)
    else:
        from kennard_stone import train_test_split as ks
        return ks(X, y, test_size=split, random_state=42)
    
# Function to standardize data
@st.cache_data
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


# Function to generate model statistics
def generate_model_statistics(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
    matthews_corrcoef = metrics.matthews_corrcoef(y_true, y_pred)
    result_list = [accuracy, precision, recall, f1_score, matthews_corrcoef]
    return result_list


# Function to generate confusion matrix plot
@st.cache_data
def generate_confusion_matrix_plot(y_train, preds, y_test, y_pred):
    # Confusion matrix (train)
    cf_matrix_train = confusion_matrix(y_train, preds)
    group_names = ['True negative', 'False positive', 'False negative', 'True positive']
    group_counts_train = ["{0:0.0f}".format(value) for value in cf_matrix_train.flatten()]
    group_percentages_train = ["{0:.0%}".format(value) for value in cf_matrix_train.flatten() / np.sum(cf_matrix_train)]
    labels_train = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_percentages_train, group_counts_train)]
    labels_train = np.asarray(labels_train).reshape(2, 2)

    # Confusion matrix (valid)
    cf_matrix_valid = confusion_matrix(y_test, y_pred)
    group_counts_valid = ["{0:0.0f}".format(value) for value in cf_matrix_valid.flatten()]
    group_percentages_valid = ["{0:.0%}".format(value) for value in cf_matrix_valid.flatten() / np.sum(cf_matrix_valid)]
    labels_valid = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_percentages_valid, group_counts_valid)]
    labels_valid = np.asarray(labels_valid).reshape(2, 2)


    return cf_matrix_train, cf_matrix_valid, labels_train, labels_valid

@st.cache_data
def download_excel_files(excel_buffer, df, without_NA, all_df_cat, train_set, valid_set, train_scaled_df, valid_scaled_df):
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="dataset", index=False)
        without_NA.to_excel(writer, sheet_name="without_NA_values", index=False)
        all_df_cat.to_excel(writer, sheet_name="converted_cat_variables", index=False)
        train_set.to_excel(writer, sheet_name="Training", index=False)
        valid_set.to_excel(writer, sheet_name="Validation", index=False)
        train_scaled_df.to_excel(writer, sheet_name="Training_set_scaled", index=False)
        valid_scaled_df.to_excel(writer, sheet_name="Validation_set_scaled", index=False)


@st.cache_data
def download_excel_files_PCA(excel_buffer, df, without_NA, all_df_cat, eigenvalues, loadings):
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="dataset", index=False)
        without_NA.to_excel(writer, sheet_name="without_NA_values", index=False)
        all_df_cat.to_excel(writer, sheet_name="converted_cat_variables", index=False)
        eigenvalues.to_excel(writer, sheet_name="eigenvalues", index=False)
        loadings.to_excel(writer, sheet_name="loadings", index=False)


def data_preprocessing_tab():
    st.title('Data preprocessing')
    st.markdown('##### Explore the dataset chosen by you and check how it presents')
    #st.sidebar.header("Data transfer")
    #uploaded_file = st.sidebar.file_uploader("Upload your file here...", type=['xlsx'])

    st.subheader("Sheet view")
    st.markdown(f"##### Currently Selected: `{sheet_selector}`")
    st.write(df)

    # Statistics
    st.subheader("Statistics")
    st.write("Table with basic statistics")
    data = df.describe()
    df_stat = pd.DataFrame(data)
    st.dataframe(df_stat)

    ## NA values
    st.markdown("##### Verity the number of missing values ")
    count_nan_in_df = df.isnull().sum()
    data_na = pd.DataFrame(count_nan_in_df, columns=["NA values"])
    st.dataframe(data_na)

    # Data visualization
    st.subheader("Data visualization")
    plot_options = st.radio("Which type of plot, would you like to see?", 
                            ("Bar plot", "Scatter plot","Histogram"))

    if plot_options == "Bar plot":
        variable_1 = st.selectbox('Choose the first variable:', (df.columns))
        variable_2 = st.selectbox('Choose the second variable:', (df.columns))
        fig, ax = plt.subplots()
        sns.barplot(x=df[variable_1], y=df[variable_2], ax=ax)
        sns.set_style("whitegrid")
        st.pyplot(fig)

    elif plot_options == "Scatter plot":
        variable_1 = st.selectbox('Choose the first variable:', (df.columns))
        variable_2 = st.selectbox('Choose the second variable:', (df.columns))
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[variable_1], y=df[variable_2], ax=ax)
        sns.set_style("whitegrid")
        st.pyplot(fig)

    else:
        variable_1 = st.selectbox('Choose the variable:', (df.columns))
        fig, ax = plt.subplots()
        sns.histplot(df[variable_1], ax=ax)
        sns.set_style("whitegrid")
        plt.xticks(rotation=90)
        st.pyplot(fig)


# Principal Component Analysis (PCA) function
def pca_tab():
    st.title("Principal Components Analysis (PCA)")
    st.markdown('##### You can use the PCA method to observer the relationship between variables and objects in your dataset. In addition, it is helpful in selecting descriptors for predictive models')

    st.subheader("Sheet view")
    st.markdown(f"##### Currently Selected: `{sheet_selector}`")
    st.write(df)

    # Choose descriptors to model + observed value/effect
    st.subheader('Create your dataset to analysis')

    # X variables
    X_PCA = st.multiselect('Select descriptors to build the model:', df.columns, key="x_multiselect_PCA")

    # Add a flag to control the flow of execution
    execute_pca = True

    # Combine selected variables
    df_PCA = pd.DataFrame(df[X_PCA])
    st.markdown('##### Your dataset to analysis:')
    st.write(df_PCA)

    # Removing NA values
    without_NA_PCA = df_PCA.dropna(axis=0, how="any")
    st.markdown(" ")
    st.markdown('##### Your dataset after removing NA values looks like this:')

    if st.button('View dataset', key='button_NA_PCA'):
        st.write(without_NA_PCA)
    else:
        st.write(' ')


    # Categorical variables
    st.subheader('Categorical variables converting')
    cat_variables_PCA = st.multiselect('Choose categorical variable(s) to separate descriptors:', without_NA_PCA.columns, key="cat_var_PCA")

    if cat_variables_PCA:
        df_2_PCA = separate_categorical(without_NA_PCA, cat_variables_PCA)
        st.markdown(" ")
        st.markdown('##### Your dataset with categorical variables separated looks like this:')
        st.write(df_2_PCA)
    else:
        st.markdown('##### You have not selected any categorical variable to separate. Your dataset remains unchanged:')
        df_2_PCA = without_NA_PCA
        st.write(df_2_PCA)


    # Instead of error, show messages below  
    if not X_PCA:
        st.warning("Please select descriptors to build the model.")
        execute_pca = False
    elif not all(df_2_PCA.dtypes.apply(lambda x: np.issubdtype(x, np.number) or np.issubdtype(x, np.bool_))):
        st.warning("Ensure all selected descriptors are in numeric or boolean format.")
        execute_pca = False
    else:
        # Standardization
        st.subheader("Standardize your data")
        st.markdown("Standardize the data - don't worry, I'll do it for you!")
        df_st = StandardScaler().fit_transform(df_2_PCA)
        df_std = pd.DataFrame(df_st, index=df_2_PCA.index, columns=df_2_PCA.columns)
        st.markdown("##### Your standardized dataset")
        st.write(df_std)

    # PCA analysis
    if execute_pca:
        st.subheader("Principal Component Analysis, finally!")

        pca = PCA()
        pca.fit(df_st)
        pca_data = pca.transform(df_st)

        st.markdown("##### Eigenvalues and explained variance")

        # Explained variance
        per_var = np.round(pca.explained_variance_ratio_.round(4)*100, decimals=1) 
        var = np.round(pca.explained_variance_.round(4), decimals=1)

        labels = ['PC' + str(x) for x in range (1, len(per_var)+1)]

        explained_var = pd.DataFrame(per_var, index=labels, columns=['Explained Variance [%]'])
        eigenvalues = pd.DataFrame(var, index=labels,  columns=['Eigenvalues'])

        scores = pd.DataFrame(pca_data, index=df_2_PCA.index, columns=labels)

        # Plot
        sns.set_style('whitegrid')
        csfont = {'fontname':'Trebuchet MS'}
        hfont = {'fontname':'Verdana'}

        fig, ax = plt.subplots(figsize=(25, 12))
        pps = ax.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels, color='lightblue')

        ax.set_xticks(range(1, len(per_var) + 1))
        ax.set_xticklabels(labels, fontsize=12, **hfont)
        ax.set_yticklabels(ax.get_yticks(), fontsize=14, **hfont)
        ax.set_ylabel("Explained Variance [%]", fontsize=18, **csfont, labelpad=20)
        ax.set_xlabel("Principal Components", fontsize=18, **csfont, labelpad=20)
        st.pyplot(fig)

        # Eigenvalues table
        st.markdown('##### Look at eigenvalues table with explained variance!')
        eigenvalues_table = explained_var.join(eigenvalues)
        st.write(eigenvalues_table)


        # Correlation matrix
        st.markdown("##### Correlation Heatmap")

        plt.figure(figsize=(14, 12))
        heatmap = sns.heatmap(df_std.corr(), vmin=-1, vmax=1, annot=True, cmap='Blues_r')
        heatmap.set_title('Correlation matrix', fontdict={'fontsize': 25}, pad=14)
        plt.xticks(rotation=55, ha='right', rotation_mode='anchor', fontsize=12)
        plt.yticks(fontsize=12)

        st.pyplot(plt)

        # Factor loadings 
        st.markdown("##### Loadings")

        # Calculate the loadings values
        loadings = pca.components_
        num_pc = pca.n_features_
        pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
        loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
        loadings_df['Variables'] = df_2_PCA.columns.values
        loadings_df = loadings_df.set_index('Variables')
        st.write(loadings_df)


        # Factor loadings plot
        plt.figure(figsize=(45, 12))
        csfont = {'fontname': 'Trebuchet MS'}
        hfont = {'fontname': 'Verdana'}

        # X variables
        PCs_choice = st.multiselect('Choose what PCs you want to see:', loadings_df.columns, key="PCs_choice")

        plt.plot(loadings_df.loc[:, PCs_choice])
        plt.xticks(rotation=50, ha='right', rotation_mode='anchor', fontsize=20, **csfont)
        plt.yticks(fontsize=20, **csfont)
        plt.xlabel("Variables", fontsize=30, **csfont)
        plt.ylabel("Loadings", fontsize=30, **csfont, labelpad=20)
        plt.legend(loadings_df.columns, loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.axhline(-0.7, c='gray', linestyle='--')
        plt.axhline(0.7, c='gray', linestyle='--')
        st.pyplot(plt)

        # Biplot
        st.subheader('Explore the principal components space')

        col1, col2 = st.columns(2)
        
        with col1: 
            x_ax = st.selectbox('Select the PC as the x axis', scores.columns)
            y_ax = st.selectbox('Select the PC as the y axis:', scores.columns)

        with col2:
            color_pca = st.selectbox('Select the variable by which you want to color the objects:', df_2_PCA.columns, key='color_PCA')
            size_pca = st.selectbox('Select the variable by which you want to assign the objects:', df_2_PCA.columns, key='size_PCA')

        fig, ax = plt.subplots()
        sns.scatterplot(x=scores[x_ax], y=scores[y_ax], ax=ax, hue=df_2_PCA[color_pca], size=df_2_PCA[size_pca])
        sns.set_style('whitegrid')
        plt.axhline(0.0, c='gray', linestyle='--')
        plt.axvline(0.0, c='gray', linestyle='--')
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        st.pyplot(fig)

        
        st.subheader('Download all excel files and report!')
            # Function to generate a PDF report
        def generate_report_PCA():
            doc = SimpleDocTemplate("PCA_report.pdf", pagesize=letter)
            story = []

            # Header
            styles = getSampleStyleSheet()
            story.append(Paragraph("PCA Report", styles["Title"]))
            story.append(Spacer(1, 12))

            # Scree Plot
            story.append(Paragraph("Scree Plot", styles["Heading3"]))
            img_buffer = BytesIO()
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(per_var) + 1), per_var, tick_label=labels, color='lightblue')
            plt.xlabel("Principal Components")
            plt.ylabel("Explained Variance [%]")
            plt.title("Scree Plot")
            plt.savefig(img_buffer, format="png")
            plt.close()
            img_buffer.seek(0)
            img = Image(img_buffer, width=500, height=300)
            story.append(img)
            story.append(Spacer(1, 6))

            # Correlation Matrix and Heatmap
            story.append(Paragraph("Correlation Matrix and Heatmap", styles["Heading3"]))
            img_buffer = BytesIO()
            plt.figure(figsize=(8, 6))
            sns.heatmap(df_std.corr(), vmin=-1, vmax=1, annot=True, cmap='Blues_r')
            plt.title("Correlation Heatmap")
            plt.savefig(img_buffer, format="png")
            plt.close()
            img_buffer.seek(0)
            img = Image(img_buffer, width=450, height=350)
            story.append(img)
            story.append(Spacer(1, 6))

            # Factor Loadings Plot
            story.append(Paragraph("Factor Loadings Plot", styles["Heading3"]))
            img_buffer = BytesIO()
            plt.figure(figsize=(12, 6))
            plt.plot(loadings_df.loc[:, PCs_choice])
            plt.xlabel("Variables")
            plt.ylabel("Loadings")
            plt.title("Factor Loadings Plot")
            plt.legend(loadings_df.columns, loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.axhline(-0.7, c='gray', linestyle='--')
            plt.axhline(0.7, c='gray', linestyle='--')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(img_buffer, format="png")
            plt.close()
            img_buffer.seek(0)
            img = Image(img_buffer, width=450, height=300)
            story.append(img)
            story.append(Spacer(1, 6))

            # Biplot
            story.append(Paragraph("Biplot", styles["Heading3"]))
            img_buffer = BytesIO()
            fig, ax = plt.subplots()
            sns.scatterplot(x=scores[x_ax], y=scores[y_ax], ax=ax, hue=df_2_PCA[color_pca], size=df_2_PCA[size_pca])
            sns.set_style("whitegrid")
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.axhline(0.0, c='gray', linestyle='--')
            plt.axvline(0.0, c='gray', linestyle='--')
            plt.title("Biplot")
            plt.tight_layout()
            plt.savefig(img_buffer, format="png")
            plt.close()
            img_buffer.seek(0)
            img = Image(img_buffer, width=450, height=330)
            story.append(img)
            story.append(Spacer(1, 6))

            # Build the PDF document
            doc.build(story)


        # Generate and download report
        generate_report_PCA()

        ## Download generated report
        st.download_button(
            label="Download Report",
            data=open("PCA_report.pdf", "rb").read(),
            file_name="PCA_report.pdf",
            mime='application/pdf',
            key="button_report_PCA"
        )



        # Download excel file
        excel_buffer = BytesIO()
        excel_files = download_excel_files_PCA(excel_buffer, df, without_NA_PCA, df_2_PCA, eigenvalues_table, loadings_df)
        st.download_button(
            label='Download Excel',
            data=excel_buffer,
            file_name='PCA_excel.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key="button_excel_PCA"
        )


# K Nearest Neighbour (KNN) function
def knn_tab():
    st.title("K-Nearest Neighbour method (k-NN)")
    st.markdown('##### Make a predictive model using the k-NN method')

    # Choose descriptors to model + observed value/effect
    st.subheader('Create your dataset to modeling')

    # X variables
    X_KNN = st.multiselect('Select descriptors to build the model:', df.columns, key="x_multiselect_KNN")

    # y variable
    y_KNN = st.multiselect('Select the property you would like to model:', df.columns, key="y_multiselect_KNN")

    # Add a flag to control the flow of execution
    execute_knn = True

    # Combine selected variables
    df_KNN = create_dataset(df, X_KNN, y_KNN)
    st.markdown('##### The dataset created by you:')
    st.write(df_KNN)


    # Removing NA values
    without_NA_KNN = df_KNN.dropna(axis=0, how="any")
    st.markdown(" ")
    st.markdown('##### Your dataset after removing NA values looks like this:')

    if st.button('View dataset', key='button_NA_KNN'):
        st.write(without_NA_KNN)
    else:
        st.write(' ')

    # Usuwamy wybrane kolumny 'X' i 'y' z całego zbioru danych 'without_NA'
    without_NA_X_KNN = without_NA_KNN.drop(columns=y_KNN)
    without_NA_y_KNN = without_NA_KNN.drop(columns=X_KNN)


    # Categorical variables
    st.subheader('Categorical variables converting')
    cat_variables_KNN = st.multiselect('Choose categorical variable(s) to separate descriptors:', without_NA_X_KNN.columns, key="cat_var_KNN")

    if cat_variables_KNN:
        X_df_2_KNN = separate_categorical(without_NA_X_KNN, cat_variables_KNN)
        st.markdown(" ")
        st.markdown('##### Your dataset with categorical variables separated looks like this:')
        all_df_KNN = X_df_2_KNN.join(without_NA_y_KNN)
        st.write(all_df_KNN)
    else:
        st.markdown('##### You have not selected any categorical variable to separate. Your dataset remains unchanged:')
        X_df_2_KNN = without_NA_X_KNN
        all_df_KNN = X_df_2_KNN.join(without_NA_y_KNN)
        st.write(all_df_KNN)


    # Data splitting (train and test sets)
    st.subheader("It's time to split your data for traning and validation sets!")
    # Choose a method & split data

    col1, col2 = st.columns(2)

    with col1:
        method_KNN = st.radio("###### Which method do you choose?", 
                                ("Scikit-learn (random)", "Kennard Stone"), key="method_split_KNN")
    
    with col2:
        split_KNN = (st.slider("###### How much of the harvest will you devote to testing the model? [%]", min_value=10, max_value=40, key="split_KNN"))/100



    # Splitting
    if method_KNN == "Scikit-learn (random)":
        X_train_KNN, X_test_KNN, y_train_KNN, y_test_KNN = split_data(X_df_2_KNN, without_NA_y_KNN, method_KNN, split_KNN)
    else:
        X_train_KNN, X_test_KNN, y_train_KNN, y_test_KNN = split_data(X_df_2_KNN, without_NA_y_KNN, method_KNN, split_KNN)


    st.markdown(' ')
    st.subheader("View your training and validation sets")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Training Set")
        train_set_KNN = X_train_KNN.join(y_train_KNN)
        if st.button('View Training Set', key="button_train_set_KNN"):
            st.write(train_set_KNN)
        else:
            st.write(' ')
    with col2:
        st.markdown("##### Validation Set")
        valid_set_KNN = X_test_KNN.join(y_test_KNN)
        if st.button('View Validation Set', key="button_valid_set_KNN"):
            st.write(valid_set_KNN)
        else:
            st.write(' ')


    # Instead of error, show messages below  
    if not X_KNN:
        st.warning("Please select descriptors to build the model.")
        execute_knn = False
    elif not y_KNN:
        st.warning("Please select the property you'd like to model.")
        execute_knn = False
    elif not all(X_df_2_KNN.dtypes.apply(lambda x: np.issubdtype(x, np.number) or np.issubdtype(x, np.bool_))):
        st.warning("Ensure all selected descriptors are either in numeric or boolean format.")
        execute_knn = False


    if execute_knn:
        # Standardization
        st.subheader("Standardize your data")
        st.markdown("Standardize the data for better model performance.")
        X_train_KNN_std, X_test_KNN_std = standardize_data(X_train_KNN, X_test_KNN)

        st.markdown("##### Training Set (Standardized)")
        train_scaled_KNN_df = pd.DataFrame(X_train_KNN_std)
        st.write(train_scaled_KNN_df)

        st.markdown("##### Validation Set (Standardized)")
        valid_scaled_KNN_df = pd.DataFrame(X_test_KNN_std)
        st.write(valid_scaled_KNN_df)


        # Select model hyperparameters (GridSearchCV or mannual)
        st.subheader("It's time to choose hyperparameter your model!")
        hyper_method_KNN = st.radio("###### How would you like to select hyperparameters?", 
                                ("GridSearchCV", "Yourself"), key="hyperparm_KNN")

        # GridSearchCV
        if hyper_method_KNN == "GridSearchCV":
            param_grid = {'n_neighbors':list(range(2,10)),
                        'weights':['distance', 'uniform']}
            
            kNN = KNeighborsClassifier()
            grid_search_KNN = GridSearchCV(estimator=kNN, param_grid=param_grid, cv=10, verbose=True)
            grid_search_KNN.fit(X_train_KNN_std, y_train_KNN)
            st.write(f"Best score: {grid_search_KNN.best_score_:.2f} using {grid_search_KNN.best_params_}")
            #st.write("Best score: %f using %s" % (grid_search.best_score_, grid_search.best_params_))

            # kNN model by gridsearchCV
            KNN = KNeighborsClassifier(**grid_search_KNN.best_params_)
            KNN.fit(X_train_KNN_std, y_train_KNN)

        # Mannual way

        else:
            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                n_neighbors = st.slider('###### What number of neighbors do you choose?', 1, 10, 1, 1, key="n_neighbours")

            with col2:
                weights = st.radio('###### Which weight function do you choose?', ('distance', 'uniform'), key="weights")

            # kNN model by user
            KNN = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights)
            KNN.fit(X_train_KNN_std, y_train_KNN)

        # Predicted values
        y_pred_KNN = KNN.predict(X_test_KNN_std)
        y_pred_train_KNN = KNN.predict(X_train_KNN_std)
        

        # Check the quality of the model
        st.markdown(' ')
        st.subheader("Let's check how our model is doing")
        st.markdown("Below you can find calculated statistics, allowing you to assess the performance, as well as the correctness of the created predictive model.")

        ## !CHANGE: the user can change colors of confustion matrixes itself
        ## !CHANGE: generowanie grafik oddzielnie (rozne formaty), tabele oddzielnie do .xlsx/.csv

        # Statistics
        train_stats_KNN = generate_model_statistics(y_train_KNN, y_pred_train_KNN)
        valid_stats_KNN = generate_model_statistics(y_test_KNN, y_pred_KNN)

        # Plot confusion matrices side by side
        # Display confusion matrices
        st.subheader("Confusion Matrices")
        cf_matrix_train, cf_matrix_valid, labels_train, labels_valid = generate_confusion_matrix_plot(y_train_KNN, y_pred_train_KNN, y_test_KNN, y_pred_KNN)

        # Plot confusion matrices side by side
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # color selecting for train confusion matrix
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            color_train = st.color_picker('Pick a color for confusion matrix (training set)', '#5F9EA0', key="color_train")

        # color selecting for valid confusion matrix
        with col2:
            color_valid = st.color_picker('Pick a color for confusion matrix (validation set)', '#8FBC8F', key="color_valid")

        cmap = sns.light_palette(color_train, as_cmap=True)
        sns.heatmap(cf_matrix_train, annot=labels_train, fmt='',cmap=cmap, ax=ax[0])
        ax[0].set_title('Confusion matrix for training set')

        cmap = sns.light_palette(color_valid, as_cmap=True)
        sns.heatmap(cf_matrix_valid, annot=labels_valid, fmt='',cmap=cmap, ax=ax[1])
        ax[1].set_title('Confusion matrix for validation set')
        st.pyplot(fig)

        # Display model statistics
        st.subheader("Model Statistics")
        col1, col2 = st.columns([0.55,0.45])

        with col1:
            st.markdown("##### Training Set")
            st.markdown(f"Accuracy: {train_stats_KNN[0]:.2f}")
            st.markdown(f"Precision: {train_stats_KNN[1]:.2f}")
            st.markdown(f"Recall: {train_stats_KNN[2]:.2f}")
            st.markdown(f"F1 Score: {train_stats_KNN[3]:.2f}")
            st.markdown(f"MCC: {train_stats_KNN[4]:.2f}")

        with col2:
            st.markdown("##### Validation Set")
            st.markdown(f"Accuracy: {valid_stats_KNN[0]:.2f}")
            st.markdown(f"Precision: {valid_stats_KNN[1]:.2f}")
            st.markdown(f"Recall: {valid_stats_KNN[2]:.2f}")
            st.markdown(f"F1 Score: {valid_stats_KNN[3]:.2f}")
            st.markdown(f"MCC: {valid_stats_KNN[4]:.2f}")

        # Prepare to report
        counts_KNN = without_NA_y_KNN.value_counts().tolist()
        column_names_list = X_df_2_KNN.columns
        output_str_KNN = ",\n".join(column_names_list)
        split_KNN_train = 1 - split_KNN


        st.subheader('Download all excel files and report!')
        # Function to generate a PDF report
        def generate_report_KNN():

            doc = SimpleDocTemplate("KNN_model_report.pdf", pagesize=letter)
            story = []

            # Header
            styles = getSampleStyleSheet()
            story.append(Paragraph("Model Evaluation Report - KNN", styles["Title"]))
            story.append(Spacer(1, 12))

            # Data
            story.append(Paragraph("Dataset", styles["Heading3"]))
            story.append(Paragraph(f"Objects: {X_df_2_KNN.shape[0]}", styles["Normal"]))
            story.append(Paragraph(f"Descriptors: {X_df_2_KNN.shape[1]}", styles["Normal"]))
            story.append(Paragraph(f"Classes: Negative ({counts_KNN[0]}), Positive ({counts_KNN[1]})", styles["Normal"]))
            story.append(Paragraph(f"Descriptors - names: {output_str_KNN}", styles["Normal"]))
            story.append(Spacer(1, 12))
            
            # Split data
            story.append(Paragraph("Data split", styles["Heading3"]))
            story.append(Paragraph(f"Method: {method_KNN}", styles["Normal"]))
            story.append(Paragraph(f"Training set: {train_set_KNN.shape[0]} ({split_KNN_train*100:.0f}%)", styles["Normal"]))
            story.append(Paragraph(f"Testing set: {valid_set_KNN.shape[0]} ({split_KNN*100:.0f}%)", styles["Normal"]))
            story.append(Spacer(1, 12))

            # Model Hyperparameters
            story.append(Paragraph("Model Hyperparameters", styles["Heading3"]))
            story.append(Paragraph(f"Method: {hyper_method_KNN}", styles["Normal"]))
            if hyper_method_KNN == "GridSearchCV":
                story.append(Paragraph(f"Best hyperparameters: {grid_search_KNN.best_params_}", styles["Normal"]))
            else:
                story.append(Paragraph(f"n_neighbors: {n_neighbors}", styles["Normal"]))
                story.append(Paragraph(f"weights: {weights}", styles["Normal"]))
            story.append(Spacer(1, 12))



            # Confusion Matrices
            story.append(Paragraph("Confusion Matrices", styles["Heading3"]))
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            cmap = sns.light_palette(color_train, as_cmap=True)
            sns.heatmap(cf_matrix_train, annot=labels_train, fmt='', cmap=cmap, ax=ax[0])
            ax[0].set_title('Confusion matrix for training set')

            cmap = sns.light_palette(color_valid, as_cmap=True)
            sns.heatmap(cf_matrix_valid, annot=labels_valid, fmt='',cmap=cmap, ax=ax[1])
            ax[1].set_title('Confusion matrix for validation set')

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)

            buffer.seek(0)
            img = buffer.read()
            buffer.close()
            story.append(Spacer(1, 12))
            story.append(Paragraph("Confusion matrices for training and validation sets:", styles["Normal"]))
            story.append(Spacer(1, 12))
            story.append(Image(BytesIO(img), 500, 200))
            story.append(Spacer(1, 12))

            # Model Statistics
            story.append(Paragraph("Model Statistics", styles["Heading3"]))
            data = [
                ["Statistic", "Training Set", "Validation Set"],
                ["Accuracy", f"{train_stats_KNN[0]:.2f}", f"{valid_stats_KNN[0]:.2f}"],
                ["Precision", f"{train_stats_KNN[1]:.2f}", f"{valid_stats_KNN[1]:.2f}"],
                ["Recall", f"{train_stats_KNN[2]:.2f}", f"{valid_stats_KNN[2]:.2f}"],
                ["F1 Score", f"{train_stats_KNN[3]:.2f}", f"{valid_stats_KNN[3]:.2f}"],
                ["MCC", f"{train_stats_KNN[4]:.2f}", f"{valid_stats_KNN[4]:.2f}"]
            ]
            t = Table(data)
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), '#8a8a8a'),
                                ('TEXTCOLOR', (0, 0), (-1, 0), '#ffffff'),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), '#f7f7f7'),
                                ('GRID', (0, 0), (-1, -1), 1, '#a0a0a0'),
                                ]))
            story.append(t)
            story.append(Spacer(1, 12))

            doc.build(story)


        # Generate and download report
        generate_report_KNN()

        ## Download generated report
        st.download_button(
            label="Download Report",
            data=open("KNN_model_report.pdf", "rb").read(),
            file_name="KNN_model_report.pdf",
            mime='application/pdf',
            key="button_report_KNN"
        )



        # Download excel file
        excel_buffer = BytesIO()
        excel_files = download_excel_files(excel_buffer, df, without_NA_KNN, all_df_KNN, train_set_KNN, valid_set_KNN, train_scaled_KNN_df, valid_scaled_KNN_df)
        st.download_button(
            label='Download Excel',
            data=excel_buffer,
            file_name='KNN_excel.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key="button_excel_KNN"
        )


# Support Vector Machine (SVM) function
def svm_tab():
    st.title("Support Vector Machine (SVM)")
    st.markdown('##### Make a predictive model using the SVM method')

    # Choose descriptors to model + observed value/effect
    st.subheader('Create your dataset to modeling')

    # X variables
    X_SVM = st.multiselect('Select descriptors to build the model:', df.columns, key="x_multiselect_SVM")

    # y variable
    y_SVM = st.multiselect('Select the property you would like to model:', df.columns, key="y_multiselect_SVM")

    # Add a flag to control the flow of execution
    execute_svm = True

    # Combine selected variables
    df_SVM = create_dataset(df, X_SVM, y_SVM)
    st.markdown('##### The dataset created by you:')
    st.write(df_SVM)


    # Removing NA values
    without_NA_SVM = df_SVM.dropna(axis=0, how="any")
    st.markdown(" ")
    st.markdown('##### Your dataset after removing NA values looks like this:')

    if st.button('View dataset', key='button_NA_SVM'):
        st.write(without_NA_SVM)
    else:
        st.write(' ')

    # Usuwamy wybrane kolumny 'X' i 'y' z całego zbioru danych 'without_NA'
    without_NA_X_SVM = without_NA_SVM.drop(columns=y_SVM)
    without_NA_y_SVM = without_NA_SVM.drop(columns=X_SVM)


    # Categorical variables
    st.subheader('Categorical variables converting')
    cat_variables_SVM = st.multiselect('Choose categorical variable(s) to separate descriptors:', without_NA_X_SVM.columns, key="cat_var_SVM")

    if cat_variables_SVM:
        X_df_2_SVM = separate_categorical(without_NA_X_SVM, cat_variables_SVM)
        st.markdown(" ")
        st.markdown('##### Your dataset with categorical variables separated looks like this:')
        all_df_SVM = X_df_2_SVM.join(without_NA_y_SVM)
        st.write(all_df_SVM)
    else:
        st.markdown('##### You have not selected any categorical variable to separate. Your dataset remains unchanged:')
        X_df_2_SVM = without_NA_X_SVM
        all_df_SVM = X_df_2_SVM.join(without_NA_y_SVM)
        st.write(all_df_SVM)


    # Data splitting (train and test sets)
    st.subheader("It's time to split your data for traning and validation sets!")
    # Choose a method & split data

    col1, col2 = st.columns(2)

    with col1:
        method_SVM = st.radio("###### Which method do you choose?", 
                                ("Scikit-learn (random)", "Kennard Stone"), key="method_split_SVM")
    
    with col2:
        split_SVM = (st.slider("###### How much of the harvest will you devote to testing the model? [%]", min_value=10, max_value=40, key="split_SVM"))/100


    # Splitting
    if method_SVM == "Scikit-learn (random)":
        X_train_SVM, X_test_SVM, y_train_SVM, y_test_SVM = split_data(X_df_2_SVM, without_NA_y_SVM, method_SVM, split_SVM)
    else:
        X_train_SVM, X_test_SVM, y_train_SVM, y_test_SVM = split_data(X_df_2_SVM, without_NA_y_SVM, method_SVM, split_SVM)

    st.markdown(' ')
    st.subheader("View your training and validation sets")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Training Set")
        train_set_SVM = X_train_SVM.join(y_train_SVM)
        if st.button('View Training Set', key="button_train_set_SVM"):
            st.write(train_set_SVM)
        else:
            st.write(' ')
    with col2:
        st.markdown("##### Validation Set")
        valid_set_SVM = X_test_SVM.join(y_test_SVM)
        if st.button('View Validation Set', key="button_valid_set_SVM"):
            st.write(valid_set_SVM)
        else:
            st.write(' ')

    if not X_SVM:
        st.warning("Please select descriptors to build the model.")
        execute_svm = False
    elif not y_SVM:
        st.warning("Please select the property you'd like to model.")
        execute_svm = False
    elif not all(X_df_2_SVM.dtypes.apply(lambda x: np.issubdtype(x, np.number) or np.issubdtype(x, np.bool_))):
        st.warning("Ensure all selected descriptors are either in numeric or boolean format.")
        execute_svm = False

    # Instead of error, show messages below  
    if execute_svm:           
        # Standardization
        st.subheader("Standardize your data")
        st.markdown("Standardize the data for better model performance.")
        X_train_SVM_std, X_test_SVM_std = standardize_data(X_train_SVM, X_test_SVM)

        st.markdown("##### Training Set (Standardized)")
        train_scaled_SVM_df = pd.DataFrame(X_train_SVM_std)
        st.write(train_scaled_SVM_df)

        st.markdown("##### Validation Set (Standardized)")
        valid_scaled_SVM_df = pd.DataFrame(X_test_SVM_std)
        st.write(valid_scaled_SVM_df)


        # Select model hyperparameters (GridSearchCV or mannual)
        st.subheader("It's time to choose hyperparameter your model!")
        hyper_method_SVM = st.radio("###### How would you like to select hyperparameters?", 
                                ("GridSearchCV", "Yourself"), key="hyperparm_SVM")
        
        # CV = 5 na potrzebe wynikow - zmienic!
        # GridSearchCV
        if hyper_method_SVM == "GridSearchCV":
            param_grid_SVM = {'C':[0.1,1,10],
                        'kernel':['linear','rbf','poly']}
            
            svc = SVC()
            grid_search_SVM = GridSearchCV(estimator=svc, param_grid=param_grid_SVM, cv=5, verbose=True)
            grid_search_SVM.fit(X_train_SVM_std, y_train_SVM)
            st.write(f"Best score: {grid_search_SVM.best_score_:.2f} using {grid_search_SVM.best_params_}")
            #st.write("Best score: %f using %s" % (grid_search.best_score_, grid_search.best_params_))

            # SVM model by gridsearchCV
            svc = SVC(**grid_search_SVM.best_params_)
            svc.fit(X_train_SVM_std, y_train_SVM)

        # Mannual way

        else:
            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                C_parm = st.slider('###### What number of C parameter do you choose?', 1, 10, 1, 1, key="C_parm") # Regularization parameter

            with col2:
                kernel = st.radio('###### What kernel do you choose?', ('linear','rbf', 'poly', 'sigmoid'), key="kernel") # Kernel type

            # kNN model by user
            svc = SVC(C = C_parm, kernel = kernel)
            svc.fit(X_train_SVM_std, y_train_SVM)

        # Predicted values
        y_pred_SVM = svc.predict(X_test_SVM_std)
        y_pred_train_SVM = svc.predict(X_train_SVM_std)
        

        # Check the quality of the model
        st.markdown(' ')
        st.subheader("Let's check how our model is doing")
        st.markdown("Below you can find calculated statistics, allowing you to assess the performance, as well as the correctness of the created predictive model.")

        ## !CHANGE: the user can change colors of confustion matrixes itself
        ## !CHANGE: generowanie grafik oddzielnie (rozne formaty), tabele oddzielnie do .xlsx/.csv

        # Statistics
        train_stats_SVM = generate_model_statistics(y_train_SVM, y_pred_train_SVM)
        valid_stats_SVM = generate_model_statistics(y_test_SVM, y_pred_SVM)

        # Plot confusion matrices side by side
        # Display confusion matrices
        st.subheader("Confusion Matrices")
        cf_matrix_train, cf_matrix_valid, labels_train, labels_valid = generate_confusion_matrix_plot(y_train_SVM, y_pred_train_SVM, y_test_SVM, y_pred_SVM)

        # Plot confusion matrices side by side
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # color selecting for train confusion matrix
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            color_train = st.color_picker('Pick a color for confusion matrix (training set)', '#5F9EA0', key="color_train")

        # color selecting for valid confusion matrix
        with col2:
            color_valid = st.color_picker('Pick a color for confusion matrix (validation set)', '#8FBC8F', key="color_valid")

        cmap = sns.light_palette(color_train, as_cmap=True)
        sns.heatmap(cf_matrix_train, annot=labels_train, fmt='', cmap=cmap, ax=ax[0])
        ax[0].set_title('Confusion matrix for training set')

        cmap = sns.light_palette(color_valid, as_cmap=True)
        sns.heatmap(cf_matrix_valid, annot=labels_valid, fmt='', cmap=cmap, ax=ax[1])
        ax[1].set_title('Confusion matrix for validation set')
        st.pyplot(fig)

        # Display model statistics
        st.subheader("Model Statistics")
        col1, col2 = st.columns([0.55,0.45])

        with col1:
            st.markdown("##### Training Set")
            st.markdown(f"Accuracy: {train_stats_SVM[0]:.2f}")
            st.markdown(f"Precision: {train_stats_SVM[1]:.2f}")
            st.markdown(f"Recall: {train_stats_SVM[2]:.2f}")
            st.markdown(f"F1 Score: {train_stats_SVM[3]:.2f}")
            st.markdown(f"MCC: {train_stats_SVM[4]:.2f}")

        with col2:
            st.markdown("##### Validation Set")
            st.markdown(f"Accuracy: {valid_stats_SVM[0]:.2f}")
            st.markdown(f"Precision: {valid_stats_SVM[1]:.2f}")
            st.markdown(f"Recall: {valid_stats_SVM[2]:.2f}")
            st.markdown(f"F1 Score: {valid_stats_SVM[3]:.2f}")
            st.markdown(f"MCC: {valid_stats_SVM[4]:.2f}")


        # Prepare to report
        counts_SVM = without_NA_y_SVM.value_counts().tolist()
        column_names_list = X_df_2_SVM.columns
        output_str_SVM = ",\n".join(column_names_list)  # Dodaj przecinki pomiędzy nazwami deskryptorów
        split_SVM_train = 1 - split_SVM


        st.subheader('Download all excel files and report!')
        # Function to generate a PDF report
        def generate_report_SVM():

            doc = SimpleDocTemplate("SVM_model_report.pdf", pagesize=letter)
            story = []

            # Header
            styles = getSampleStyleSheet()
            story.append(Paragraph("Model Evaluation Report - SVM", styles["Title"]))
            story.append(Spacer(1, 12))

            # Data
            story.append(Paragraph("Dataset", styles["Heading3"]))
            story.append(Paragraph(f"Objects: {X_df_2_SVM.shape[0]}", styles["Normal"]))
            story.append(Paragraph(f"Descriptors: {X_df_2_SVM.shape[1]}", styles["Normal"]))
            story.append(Paragraph(f"Classes: Negative ({counts_SVM[0]}), Positive ({counts_SVM[1]})", styles["Normal"]))
            story.append(Paragraph(f"Descriptors - names: {output_str_SVM}", styles["Normal"]))
            story.append(Spacer(1, 12))
            
            # Split data
            story.append(Paragraph("Data split", styles["Heading3"]))
            story.append(Paragraph(f"Method: {method_SVM}", styles["Normal"]))
            story.append(Paragraph(f"Training set: {train_set_SVM.shape[0]} ({split_SVM_train*100:.0f}%)", styles["Normal"]))
            story.append(Paragraph(f"Testing set: {valid_set_SVM.shape[0]} ({split_SVM*100:.0f}%)", styles["Normal"]))
            story.append(Spacer(1, 12))

            # Model Hyperparameters
            story.append(Paragraph("Model Hyperparameters", styles["Heading3"]))
            story.append(Paragraph(f"Method: {hyper_method_SVM}", styles["Normal"]))
            if hyper_method_SVM == "GridSearchCV":
                story.append(Paragraph(f"Best hyperparameters: {grid_search_SVM.best_params_}", styles["Normal"]))
            else:
                story.append(Paragraph(f"C: {C_parm}", styles["Normal"]))
                story.append(Paragraph(f"kernel: {kernel}", styles["Normal"]))
            story.append(Spacer(1, 12))

            # Confusion Matrices
            story.append(Paragraph("Confusion Matrices", styles["Heading3"]))
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            cmap = sns.light_palette(color_train, as_cmap=True)
            sns.heatmap(cf_matrix_train, annot=labels_train, fmt='', cmap=cmap, ax=ax[0])
            ax[0].set_title('Confusion matrix for training set')

            cmap = sns.light_palette(color_valid, as_cmap=True)
            sns.heatmap(cf_matrix_valid, annot=labels_valid, fmt='', cmap=cmap, ax=ax[1])
            ax[1].set_title('Confusion matrix for validation set')

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)

            buffer.seek(0)
            img = buffer.read()
            buffer.close()
            story.append(Spacer(1, 12))
            story.append(Paragraph("Confusion matrices for training and validation sets:", styles["Normal"]))
            story.append(Spacer(1, 12))
            story.append(Image(BytesIO(img), 500, 200))
            story.append(Spacer(1, 12))

            # Model Statistics
            story.append(Paragraph("Model Statistics", styles["Heading3"]))
            data = [
                ["Statistic", "Training Set", "Validation Set"],
                ["Accuracy", f"{train_stats_SVM[0]:.2f}", f"{valid_stats_SVM[0]:.2f}"],
                ["Precision", f"{train_stats_SVM[1]:.2f}", f"{valid_stats_SVM[1]:.2f}"],
                ["Recall", f"{train_stats_SVM[2]:.2f}", f"{valid_stats_SVM[2]:.2f}"],
                ["F1 Score", f"{train_stats_SVM[3]:.2f}", f"{valid_stats_SVM[3]:.2f}"],
                ["MCC", f"{train_stats_SVM[4]:.2f}", f"{valid_stats_SVM[4]:.2f}"]
            ]

            t = Table(data)
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), '#8a8a8a'),
                                ('TEXTCOLOR', (0, 0), (-1, 0), '#ffffff'),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), '#f7f7f7'),
                                ('GRID', (0, 0), (-1, -1), 1, '#a0a0a0'),
                                ]))
            story.append(t)
            story.append(Spacer(1, 12))

            doc.build(story)


        # Generate and download report
        generate_report_SVM()

        ## Download generated report
        st.download_button(
            label="Download Report",
            data=open("SVM_model_report.pdf", "rb").read(),
            file_name="SVM_model_report.pdf",
            mime='application/pdf',
            key="button_report_SVM"
        )

        # Raport nie pobiera confusion matrix - poprawic!!
        # Problem z generowaniem raportu dla grid_search!
        # Prawdopodobnie najlepiej wkleić bezpośrednio funkcje generowania raportu pod każdą z metod (w kazdym tabie) - może mniej problematyczne rozwiazanie?

        # Download excel file
        excel_buffer = BytesIO()
        excel_files = download_excel_files(excel_buffer, df, without_NA_SVM, all_df_SVM, train_set_SVM, valid_set_SVM, train_scaled_SVM_df, valid_scaled_SVM_df)
        st.download_button(
            label='Download Excel',
            data=excel_buffer,
            file_name='SVM_excel.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key="button_excel_SVM"
        )


# Decision Tree Classifier (DTC) Function
def dtc_tab():

    st.title("Decision Tree Classifier (DTC)")
    st.markdown('##### Make a predictive model using the DTC method')

    # Choose descriptors to model + observed value/effect
    st.subheader('Create your dataset to modeling')

    # X variables
    X_DTC = st.multiselect('Select descriptors to build the model:', df.columns, key="x_multiselect_DTC")

    # y variable
    y_DTC = st.multiselect('Select the property you would like to model:', df.columns, key="y_multiselect_DTC")

    # Add a flag to control the flow of execution
    execute_dtc = True

    # Combine selected variables
    df_DTC = create_dataset(df, X_DTC, y_DTC)
    st.markdown('##### The dataset created by you:')
    st.write(df_DTC)

    # Removing NA values
    without_NA_DTC = df_DTC.dropna(axis=0, how="any")
    st.markdown(" ")
    st.markdown('##### Your dataset after removing NA values looks like this:')

    if st.button('View dataset', key='button_NA_DTC'):
        st.write(without_NA_DTC)
    else:
        st.write(' ')

    # Usuwamy wybrane kolumny 'X' i 'y' z całego zbioru danych 'without_NA'
    without_NA_X_DTC = without_NA_DTC.drop(columns=y_DTC)
    without_NA_y_DTC = without_NA_DTC[y_DTC]


    # Categorical variables
    st.subheader('Categorical variables converting')
    cat_variables_DTC = st.multiselect('Choose categorical variable(s) to separate descriptors:', without_NA_X_DTC.columns, key="cat_var_DTC")

    if cat_variables_DTC:
        X_df_2_DTC = separate_categorical(without_NA_X_DTC, cat_variables_DTC)
        st.markdown(" ")
        st.markdown('##### Your dataset with categorical variables separated looks like this:')
        all_df_DTC = X_df_2_DTC.join(without_NA_y_DTC)
        st.write(all_df_DTC)
    else:
        st.markdown('##### You have not selected any categorical variable to separate. Your dataset remains unchanged:')
        X_df_2_DTC = without_NA_X_DTC
        all_df_DTC = X_df_2_DTC.join(without_NA_y_DTC)
        st.write(all_df_DTC)


    # Data splitting (train and test sets)
    st.subheader("It's time to split your data for traning and validation sets!")
    # Choose a method & split data

    col1, col2 = st.columns(2)

    with col1:
        method_DTC = st.radio("###### Which method do you choose?", 
                                ("Scikit-learn (random)", "Kennard Stone"), key="method_split_DTC")
    
    with col2:
        split_DTC = (st.slider("###### How much of the harvest will you devote to testing the model? [%]", min_value=10, max_value=40, key="split_DTC"))/100


    # Splitting
    if method_DTC == "Scikit-learn (random)":
        X_train_DTC, X_test_DTC, y_train_DTC, y_test_DTC = split_data(X_df_2_DTC, without_NA_y_DTC, method_DTC, split_DTC)
    else:
        X_train_DTC, X_test_DTC, y_train_DTC, y_test_DTC = split_data(X_df_2_DTC, without_NA_y_DTC, method_DTC, split_DTC)

    st.markdown(' ')
    st.subheader("View your training and validation sets")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Training Set")
        train_set_DTC = X_train_DTC.join(y_train_DTC)
        if st.button('View Training Set', key="button_train_set_DTC"):
            st.write(train_set_DTC)
        else:
            st.write(' ')
    with col2:
        st.markdown("##### Validation Set")
        valid_set_DTC = X_test_DTC.join(y_test_DTC)
        if st.button('View Validation Set', key="button_valid_set_DTC"):
            st.write(valid_set_DTC)
        else:
            st.write(' ')

    # Instead of error, show messages below        
    if not X_DTC:
        st.warning("Please select descriptors to build the model.")
        execute_dtc = False
    elif not y_DTC:
        st.warning("Please select the property you'd like to model.")
        execute_dtc = False
    elif not all(X_df_2_DTC.dtypes.apply(lambda x: np.issubdtype(x, np.number) or np.issubdtype(x, np.bool_))):
        st.warning("Ensure all selected descriptors are either in numeric or boolean format.")
        execute_dtc = False

    if execute_dtc: 

        # Standardization
        st.subheader("Standardize your data")
        st.markdown("Standardize the data for better model performance.")
        X_train_DTC_std, X_test_DTC_std = standardize_data(X_train_DTC, X_test_DTC)

        st.markdown("##### Training Set (Standardized)")
        train_scaled_DTC_df = pd.DataFrame(X_train_DTC_std)
        st.write(train_scaled_DTC_df)

        st.markdown("##### Validation Set (Standardized)")
        valid_scaled_DTC_df = pd.DataFrame(X_test_DTC_std)
        st.write(valid_scaled_DTC_df)


        # Select model hyperparameters (GridSearchCV or mannual)
        st.subheader("It's time to choose hyperparameter your model!")
        hyper_method_DTC = st.radio("###### How would you like to select hyperparameters?", 
                                ("GridSearchCV", "Yourself"), key="hyperparm_DTC")

        # GridSearchCV
        if hyper_method_DTC == "GridSearchCV":
            param_grid_DTC = {'max_features': ['sqrt', 'log2'],
                            'max_depth' : list(range(2,6)),
                            'criterion' :['gini', 'entropy', 'log_loss'],
                            'splitter': ['best', 'random']}
            
            dtc = DecisionTreeClassifier(random_state=24)
            grid_search_DTC = GridSearchCV(estimator=dtc, param_grid=param_grid_DTC, cv=10, verbose=True)
            grid_search_DTC.fit(X_train_DTC_std, y_train_DTC)
            st.write(f"Best score: {grid_search_DTC.best_score_:.2f} using {grid_search_DTC.best_params_}")
            #st.write("Best score: %f using %s" % (grid_search.best_score_, grid_search.best_params_))

            # DTC model by gridsearchCV
            dtc = DecisionTreeClassifier(**grid_search_DTC.best_params_, random_state=24)
            dtc.fit(X_train_DTC_std, y_train_DTC)

        # Mannual way

        else:
            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                max_features_DTC = st.radio('###### Choose the number of features to condiser when looking for the best split:', ('sqrt', 'log2'), key="max_features_DTC") 
                max_depth_DTC = st.slider('###### Choose the maximum depth of the tree:', 2, 6, 1, 1, key="max_depth_DTC")

            with col2:
                criterion_DTC = st.radio('###### Choose the function to measure the quality of a split:', ('gini', 'entropy', 'log_loss'), key="criterion_DTC") 
                splitter_DTC = st.radio('###### Choose the strategy used to choose the split at each node:', ('best', 'random'), key="splitter_DTC")

            # Decision Tree model by user
            dtc = DecisionTreeClassifier(criterion=criterion_DTC, splitter=splitter_DTC, max_depth=max_depth_DTC, max_features=max_features_DTC, random_state=24)
            dtc.fit(X_train_DTC_std, y_train_DTC)

        # Predicted values
        y_pred_DTC = dtc.predict(X_test_DTC_std)
        y_pred_train_DTC = dtc.predict(X_train_DTC_std)
        

        # Check the quality of the model
        st.markdown(' ')
        st.subheader("Let's check how our model is doing")
        st.markdown("Below you can find calculated statistics, allowing you to assess the performance, as well as the correctness of the created predictive model.")

        ## !CHANGE: the user can change colors of confustion matrixes itself
        ## !CHANGE: generowanie grafik oddzielnie (rozne formaty), tabele oddzielnie do .xlsx/.csv

        # Statistics
        train_stats_DTC = generate_model_statistics(y_train_DTC, y_pred_train_DTC)
        valid_stats_DTC = generate_model_statistics(y_test_DTC, y_pred_DTC)

        # Plot confusion matrices side by side
        # Display confusion matrices
        st.subheader("Confusion Matrices")
        cf_matrix_train, cf_matrix_valid, labels_train, labels_valid = generate_confusion_matrix_plot(y_train_DTC, y_pred_train_DTC, y_test_DTC, y_pred_DTC)

        # Plot confusion matrices side by side
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # color selecting for train confusion matrix
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            color_train = st.color_picker('Pick a color for confusion matrix (training set)', '#5F9EA0', key="color_train")

        # color selecting for valid confusion matrix
        with col2:
            color_valid = st.color_picker('Pick a color for confusion matrix (validation set)', '#8FBC8F', key="color_valid")

        cmap = sns.light_palette(color_train, as_cmap=True)
        sns.heatmap(cf_matrix_train, annot=labels_train, fmt='', cmap=cmap, ax=ax[0])
        ax[0].set_title('Confusion matrix for training set')

        cmap = sns.light_palette(color_valid, as_cmap=True)
        sns.heatmap(cf_matrix_valid, annot=labels_valid, fmt='', cmap=cmap, ax=ax[1])
        ax[1].set_title('Confusion matrix for validation set')
        st.pyplot(fig)

        # Display model statistics
        st.subheader("Model Statistics")
        col1, col2 = st.columns([0.55,0.45])

        with col1:
            st.markdown("##### Training Set")
            st.markdown(f"Accuracy: {train_stats_DTC[0]:.2f}")
            st.markdown(f"Precision: {train_stats_DTC[1]:.2f}")
            st.markdown(f"Recall: {train_stats_DTC[2]:.2f}")
            st.markdown(f"F1 Score: {train_stats_DTC[3]:.2f}")
            st.markdown(f"MCC: {train_stats_DTC[4]:.2f}")

        with col2:
            st.markdown("##### Validation Set")
            st.markdown(f"Accuracy: {valid_stats_DTC[0]:.2f}")
            st.markdown(f"Precision: {valid_stats_DTC[1]:.2f}")
            st.markdown(f"Recall: {valid_stats_DTC[2]:.2f}")
            st.markdown(f"F1 Score: {valid_stats_DTC[3]:.2f}")
            st.markdown(f"MCC: {valid_stats_DTC[4]:.2f}")


        # DTC Visualization
        st.subheader("DTC visualization")
        fig = plt.figure(figsize=(15, 10))

        # Download columns names
        feature_names = X_train_DTC.columns.tolist()
        class_names = ["negative", "positive"]
        plot_tree(dtc, filled=True, feature_names=feature_names, class_names=class_names)
        plt.savefig("decision_tree.png")
        plt.close(fig) 
        st.pyplot(fig)

        # Prepare to report
        counts_DTC = without_NA_y_DTC.value_counts().tolist()
        column_names_list = X_df_2_DTC.columns
        output_str_DTC = ",\n".join(column_names_list)  # Dodaj przecinki pomiędzy nazwami deskryptorów
        split_DTC_train = 1 - split_DTC


        st.subheader('Download all excel files and report!')
        # Function to generate a PDF report
        def generate_report_DTC():

            doc = SimpleDocTemplate("DTC_model_report.pdf", pagesize=letter)
            story = []

            # Header
            styles = getSampleStyleSheet()
            story.append(Paragraph("Model Evaluation Report - DTC", styles["Title"]))
            story.append(Spacer(1, 12))

            # Data
            story.append(Paragraph("Dataset", styles["Heading3"]))
            story.append(Paragraph(f"Objects: {X_df_2_DTC.shape[0]}", styles["Normal"]))
            story.append(Paragraph(f"Descriptors: {X_df_2_DTC.shape[1]}", styles["Normal"]))
            story.append(Paragraph(f"Classes: Negative ({counts_DTC[0]}), Positive ({counts_DTC[1]})", styles["Normal"]))
            story.append(Paragraph(f"Descriptors - names: {output_str_DTC}", styles["Normal"]))
            story.append(Spacer(1, 12))
            
            # Split data
            story.append(Paragraph("Data split", styles["Heading3"]))
            story.append(Paragraph(f"Method: {method_DTC}", styles["Normal"]))
            story.append(Paragraph(f"Training set: {train_set_DTC.shape[0]} ({split_DTC_train*100:.0f}%)", styles["Normal"]))
            story.append(Paragraph(f"Testing set: {valid_set_DTC.shape[0]} ({split_DTC*100:.0f}%)", styles["Normal"]))
            story.append(Spacer(1, 12))

            # Model Hyperparameters
            story.append(Paragraph("Model Hyperparameters", styles["Heading3"]))
            story.append(Paragraph(f"Method: {hyper_method_DTC}", styles["Normal"]))
            if hyper_method_DTC == "GridSearchCV":
                story.append(Paragraph(f"Best hyperparameters: {grid_search_DTC.best_params_}", styles["Normal"]))
            else:
                story.append(Paragraph(f"criterion: {criterion_DTC}", styles["Normal"]))
                story.append(Paragraph(f"splitter: {splitter_DTC}", styles["Normal"]))
                story.append(Paragraph(f"max_depth: {max_depth_DTC}", styles["Normal"]))
                story.append(Paragraph(f"max_features: {max_features_DTC}", styles["Normal"]))
            story.append(Spacer(1, 12))

            # Confusion Matrices
            story.append(Paragraph("Confusion Matrices", styles["Heading3"]))
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            cmap = sns.light_palette(color_train, as_cmap=True)
            sns.heatmap(cf_matrix_train, annot=labels_train, fmt='', vmax=80, cmap=cmap, ax=ax[0])
            ax[0].set_title('Confusion matrix for training set')

            cmap = sns.light_palette(color_valid, as_cmap=True)
            sns.heatmap(cf_matrix_valid, annot=labels_valid, fmt='', vmax=32, cmap=cmap, ax=ax[1])
            ax[1].set_title('Confusion matrix for validation set')

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)

            buffer.seek(0)
            img = buffer.read()
            buffer.close()
            story.append(Spacer(1, 12))
            story.append(Paragraph("Confusion matrices for training and validation sets:", styles["Normal"]))
            story.append(Spacer(1, 12))
            story.append(Image(BytesIO(img), 500, 200))
            story.append(Spacer(1, 12))

            # Model Statistics
            story.append(Paragraph("Model Statistics", styles["Heading3"]))
            data = [
                ["Statistic", "Training Set", "Validation Set"],
                ["Accuracy", f"{train_stats_DTC[0]:.2f}", f"{valid_stats_DTC[0]:.2f}"],
                ["Precision", f"{train_stats_DTC[1]:.2f}", f"{valid_stats_DTC[1]:.2f}"],
                ["Recall", f"{train_stats_DTC[2]:.2f}", f"{valid_stats_DTC[2]:.2f}"],
                ["F1 Score", f"{train_stats_DTC[3]:.2f}", f"{valid_stats_DTC[3]:.2f}"],
                ["MCC", f"{train_stats_DTC[4]:.2f}", f"{valid_stats_DTC[4]:.2f}"]
            ]

            t = Table(data)
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), '#8a8a8a'),
                                ('TEXTCOLOR', (0, 0), (-1, 0), '#ffffff'),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), '#f7f7f7'),
                                ('GRID', (0, 0), (-1, -1), 1, '#a0a0a0'),
                                ]))
            story.append(t)
            story.append(Spacer(1, 12))

            # Utwórz wykres zapisany jako plik PNG
            story.append(Spacer(1, 12))
            story.append(Paragraph("Decision Tree:", styles["Heading3"]))
            story.append(Spacer(1, 12))
            story.append(Image("decision_tree.png", 500, 300))  # Dodaj wykres do raportu
            story.append(Spacer(1, 12))

            doc.build(story)


        # Generate and download report
        generate_report_DTC()

        ## Download generated report
        st.download_button(
            label="Download Report",
            data=open("DTC_model_report.pdf", "rb").read(),
            file_name="DTC_model_report.pdf",
            mime='application/pdf',
            key="button_report_DTC"
        )


        # Download excel file
        excel_buffer = BytesIO()
        excel_files = download_excel_files(excel_buffer, df, without_NA_DTC, all_df_DTC, train_set_DTC, valid_set_DTC, train_scaled_DTC_df, valid_scaled_DTC_df)
        st.download_button(
            label='Download Excel',
            data=excel_buffer,
            file_name='DTC_excel.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key="button_excel_DTC"
        )


# Random Forest Classifier (RFC) function
def rfc_tab():

    st.title("Random Forest Classifier (RFC)")
    st.markdown('##### Make a predictive model using the DTC method')

    # Choose descriptors to model + observed value/effect
    st.subheader('Create your dataset to modeling')

    # X variables
    X_RFC = st.multiselect('Select descriptors to build the model:', df.columns, key="x_multiselect_RFC")

    # y variable
    y_RFC = st.multiselect('Select the property you would like to model:', df.columns, key="y_multiselect_RFC")

    # Add a flag to control the flow of execution
    execute_rfc = True

    # Combine selected variables
    df_RFC = create_dataset(df, X_RFC, y_RFC)
    st.markdown('##### The dataset created by you:')
    st.write(df_RFC)


    # Removing NA values
    without_NA_RFC = df_RFC.dropna(axis=0, how="any")
    st.markdown(" ")
    st.markdown('##### Your dataset after removing NA values looks like this:')

    if st.button('View dataset', key='button_NA_RFC'):
        st.write(without_NA_RFC)
    else:
        st.write(' ')

    # Usuwamy wybrane kolumny 'X' i 'y' z całego zbioru danych 'without_NA'
    without_NA_X_RFC = without_NA_RFC.drop(columns=y_RFC)
    without_NA_y_RFC = without_NA_RFC[y_RFC]


    # Categorical variables
    st.subheader('Categorical variables converting')
    cat_variables_RFC = st.multiselect('Choose categorical variable(s) to separate descriptors:', without_NA_X_RFC.columns, key="cat_var_RFC")

    if cat_variables_RFC:
        X_df_2_RFC = separate_categorical(without_NA_X_RFC, cat_variables_RFC)
        st.markdown(" ")
        st.markdown('##### Your dataset with categorical variables separated looks like this:')
        all_df_RFC = X_df_2_RFC.join(without_NA_y_RFC)
        st.write(all_df_RFC)
    else:
        st.markdown('##### You have not selected any categorical variable to separate. Your dataset remains unchanged:')
        X_df_2_RFC = without_NA_X_RFC
        all_df_RFC = X_df_2_RFC.join(without_NA_y_RFC)
        st.write(all_df_RFC)


    # Data splitting (train and test sets)
    st.subheader("It's time to split your data for traning and validation sets!")
    # Choose a method & split data

    col1, col2 = st.columns(2)

    with col1:
        method_RFC = st.radio("###### Which method do you choose?", 
                                ("Scikit-learn (random)", "Kennard Stone"), key="method_split_RFC")
    
    with col2:
        split_RFC = (st.slider("###### How much of the harvest will you devote to testing the model? [%]", min_value=10, max_value=40, key="split_RFC"))/100


    # Splitting
    if method_RFC == "Scikit-learn (random)":
        X_train_RFC, X_test_RFC, y_train_RFC, y_test_RFC = split_data(X_df_2_RFC, without_NA_y_RFC, method_RFC, split_RFC)
    else:
        X_train_RFC, X_test_RFC, y_train_RFC, y_test_RFC = split_data(X_df_2_RFC, without_NA_y_RFC, method_RFC, split_RFC)

    st.markdown(' ')
    st.subheader("View your training and validation sets")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Training Set")
        train_set_RFC = X_train_RFC.join(y_train_RFC)
        if st.button('View Training Set', key="button_train_set_RFC"):
            st.write(train_set_RFC)
        else:
            st.write(' ')
    with col2:
        st.markdown("##### Validation Set")
        valid_set_RFC = X_test_RFC.join(y_test_RFC)
        if st.button('View Validation Set', key="button_valid_set_RFC"):
            st.write(valid_set_RFC)
        else:
            st.write(' ')

    # Instead of error, show messages below  
    if not X_RFC:
        st.warning("Please select descriptors to build the model.")
        execute_rfc = False
    elif not y_RFC:
        st.warning("Please select the property you'd like to model.")
        execute_rfc = False
    elif not all(X_df_2_RFC.dtypes.apply(lambda x: np.issubdtype(x, np.number) or np.issubdtype(x, np.bool_))):
        st.warning("Ensure all selected descriptors are either in numeric or boolean format.")
        execute_rfc = False

    if execute_rfc:             

        # Standardization
        st.subheader("Standardize your data")
        st.markdown("Standardize the data for better model performance.")
        X_train_RFC_std, X_test_RFC_std = standardize_data(X_train_RFC, X_test_RFC)

        st.markdown("##### Training Set (Standardized)")
        train_scaled_RFC_df = pd.DataFrame(X_train_RFC_std)
        st.write(train_scaled_RFC_df)

        st.markdown("##### Validation Set (Standardized)")
        valid_scaled_RFC_df = pd.DataFrame(X_test_RFC_std)
        st.write(valid_scaled_RFC_df)


        # Select model hyperparameters (GridSearchCV or mannual)
        st.subheader("It's time to choose hyperparameter your model!")
        hyper_method_RFC = st.radio("###### How would you like to select hyperparameters?", 
                                ("GridSearchCV", "Yourself"), key="hyperparm_RFC")

        # GridSearchCV
        if hyper_method_RFC == "GridSearchCV":
            param_grid_RFC = {'n_estimators': [50, 100, 200],
                            'max_features': ['sqrt', 'log2'],
                            'max_depth' : [2,3,4],
                            'criterion' :['gini', 'entropy']}
            
            rfc = RandomForestClassifier(random_state=24)
            grid_search_RFC = GridSearchCV(estimator=rfc, param_grid=param_grid_RFC, cv=10, verbose=True)
            grid_search_RFC.fit(X_train_RFC_std, y_train_RFC)
            st.write(f"Best score: {grid_search_RFC.best_score_:.2f} using {grid_search_RFC.best_params_}")
            #st.write("Best score: %f using %s" % (grid_search.best_score_, grid_search.best_params_))

            # DTC model by gridsearchCV
            rfc = RandomForestClassifier(**grid_search_RFC.best_params_, random_state=24)
            rfc.fit(X_train_RFC_std, y_train_RFC)


        # Mannual way
        else:
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                max_features_RFC = st.radio('###### Choose the number of features to condiser when looking for the best split:', ('sqrt', 'log2'), key="max_features_RFC") 
                max_depth_RFC = st.slider('###### Choose the maximum depth of the tree:', 2, 6, 1, 1, key="max_depth_RFC")

            with col2:
                criterion_RFC = st.radio('###### Choose the function to measure the quality of a split:', ('gini', 'entropy'), key="criterion_RFC") 
                n_estimators_RFC = st.number_input('Insert a number', 50, 200, key="n_estimators_RFC")

            # Decision Tree model by user
            rfc = RandomForestClassifier(criterion=criterion_RFC, n_estimators=n_estimators_RFC, max_depth=max_depth_RFC, max_features=max_features_RFC, random_state=24)
            rfc.fit(X_train_RFC_std, y_train_RFC)

        # Predicted values
        y_pred_RFC = rfc.predict(X_test_RFC_std)
        y_pred_train_RFC = rfc.predict(X_train_RFC_std)


        # Feature importance
        st.markdown(' ')
        st.subheader("Feature importance")

        # Assuming X_train is your DataFrame with the column names
        column_names = X_df_2_RFC.columns.tolist()
        fig, ax = plt.subplots()
        ax.bar(column_names, rfc.feature_importances_, color='lightskyblue')
        sns.set_style("whitegrid")
        plt.xticks(rotation=90) 
        st.pyplot(fig)
        

        # Check the quality of the model
        st.markdown(' ')
        st.subheader("Let's check how our model is doing")
        st.markdown("Below you can find calculated statistics, allowing you to assess the performance, as well as the correctness of the created predictive model.")


        # Statistics
        train_stats_RFC = generate_model_statistics(y_train_RFC, y_pred_train_RFC)
        valid_stats_RFC = generate_model_statistics(y_test_RFC, y_pred_RFC)

        # Plot confusion matrices side by side
        # Display confusion matrices
        st.subheader("Confusion Matrices")
        cf_matrix_train, cf_matrix_valid, labels_train, labels_valid = generate_confusion_matrix_plot(y_train_RFC, y_pred_train_RFC, y_test_RFC, y_pred_RFC)

        # Plot confusion matrices side by side
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # color selecting for train confusion matrix
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            color_train = st.color_picker('Pick a color for confusion matrix (training set)', '#5F9EA0', key="color_train")

        # color selecting for valid confusion matrix
        with col2:
            color_valid = st.color_picker('Pick a color for confusion matrix (validation set)', '#8FBC8F', key="color_valid")

        cmap = sns.light_palette(color_train, as_cmap=True)
        sns.heatmap(cf_matrix_train, annot=labels_train, fmt='', cmap=cmap, ax=ax[0])
        ax[0].set_title('Confusion matrix for training set')

        cmap = sns.light_palette(color_valid, as_cmap=True)
        sns.heatmap(cf_matrix_valid, annot=labels_valid, fmt='', cmap=cmap, ax=ax[1])
        ax[1].set_title('Confusion matrix for validation set')
        st.pyplot(fig)

        # Display model statistics
        st.subheader("Model Statistics")
        col1, col2 = st.columns([0.55,0.45])

        with col1:
            st.markdown("##### Training Set")
            st.markdown(f"Accuracy: {train_stats_RFC[0]:.2f}")
            st.markdown(f"Precision: {train_stats_RFC[1]:.2f}")
            st.markdown(f"Recall: {train_stats_RFC[2]:.2f}")
            st.markdown(f"F1 Score: {train_stats_RFC[3]:.2f}")
            st.markdown(f"MCC: {train_stats_RFC[4]:.2f}")

        with col2:
            st.markdown("##### Validation Set")
            st.markdown(f"Accuracy: {valid_stats_RFC[0]:.2f}")
            st.markdown(f"Precision: {valid_stats_RFC[1]:.2f}")
            st.markdown(f"Recall: {valid_stats_RFC[2]:.2f}")
            st.markdown(f"F1 Score: {valid_stats_RFC[3]:.2f}")
            st.markdown(f"MCC: {valid_stats_RFC[4]:.2f}")

        # Prepare to report
        counts_RFC = without_NA_y_RFC.value_counts().tolist()
        column_names_list = X_df_2_RFC.columns
        output_str_RFC = ",\n".join(column_names_list)  # Dodaj przecinki pomiędzy nazwami deskryptorów
        split_RFC_train = 1 - split_RFC


        st.subheader('Download all excel files and report!')
        # Function to generate a PDF report
        def generate_report_RFC():

            doc = SimpleDocTemplate("RFC_model_report.pdf", pagesize=letter)
            story = []

            # Header
            styles = getSampleStyleSheet()
            story.append(Paragraph("Model Evaluation Report - RFC", styles["Title"]))
            story.append(Spacer(1, 12))

            # Data
            story.append(Paragraph("Dataset", styles["Heading3"]))
            story.append(Paragraph(f"Objects: {X_df_2_RFC.shape[0]}", styles["Normal"]))
            story.append(Paragraph(f"Descriptors: {X_df_2_RFC.shape[1]}", styles["Normal"]))
            story.append(Paragraph(f"Classes: Negative ({counts_RFC[0]}), Positive ({counts_RFC[1]})", styles["Normal"]))
            story.append(Paragraph(f"Descriptors - names: {output_str_RFC}", styles["Normal"]))
            story.append(Spacer(1, 12))
            
            # Split data
            story.append(Paragraph("Data split", styles["Heading3"]))
            story.append(Paragraph(f"Method: {method_RFC}", styles["Normal"]))
            if hyper_method_RFC == "GridSearchCV":
                story.append(Paragraph(f"Best hyperparameters: {grid_search_RFC.best_params_}", styles["Normal"]))
            else:
                story.append(Paragraph(f"criterion: {criterion_RFC}", styles["Normal"]))
                story.append(Paragraph(f"n_estimators: {n_estimators_RFC}", styles["Normal"]))
                story.append(Paragraph(f"max_depth: {max_depth_RFC}", styles["Normal"]))
                story.append(Paragraph(f"max_features: {max_features_RFC}", styles["Normal"]))
            story.append(Spacer(1, 12))

            # Confusion Matrices
            story.append(Paragraph("Confusion Matrices", styles["Heading3"]))
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            cmap = sns.light_palette(color_train, as_cmap=True)
            sns.heatmap(cf_matrix_train, annot=labels_train, fmt='', cmap=cmap, ax=ax[0])
            ax[0].set_title('Confusion matrix for training set')

            cmap = sns.light_palette(color_valid, as_cmap=True)
            sns.heatmap(cf_matrix_valid, annot=labels_valid, fmt='', cmap=cmap, ax=ax[1])
            ax[1].set_title('Confusion matrix for validation set')

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)

            buffer.seek(0)
            img = buffer.read()
            buffer.close()
            story.append(Spacer(1, 12))
            story.append(Paragraph("Confusion matrices for training and validation sets:", styles["Normal"]))
            story.append(Spacer(1, 12))
            story.append(Image(BytesIO(img), 500, 200))
            story.append(Spacer(1, 12))

            # Model Statistics
            story.append(Paragraph("Model Statistics", styles["Heading3"]))
            data = [
                ["Statistic", "Training Set", "Validation Set"],
                ["Accuracy", f"{train_stats_RFC[0]:.2f}", f"{valid_stats_RFC[0]:.2f}"],
                ["Precision", f"{train_stats_RFC[1]:.2f}", f"{valid_stats_RFC[1]:.2f}"],
                ["Recall", f"{train_stats_RFC[2]:.2f}", f"{valid_stats_RFC[2]:.2f}"],
                ["F1 Score", f"{train_stats_RFC[3]:.2f}", f"{valid_stats_RFC[3]:.2f}"],
                ["MCC", f"{train_stats_RFC[4]:.2f}", f"{valid_stats_RFC[4]:.2f}"]
            ]

            t = Table(data)
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), '#8a8a8a'),
                                ('TEXTCOLOR', (0, 0), (-1, 0), '#ffffff'),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), '#f7f7f7'),
                                ('GRID', (0, 0), (-1, -1), 1, '#a0a0a0'),
                                ]))
            story.append(t)
            story.append(Spacer(1, 12))

            # Feature Importance
            st.markdown(' ')
            fig, ax = plt.subplots()
            sns.set_style("whitegrid")
            ax.bar(column_names, rfc.feature_importances_, color='lightskyblue')
            plt.xlabel('Features')
            plt.ylabel('Feature Importance')
            plt.title('Feature Importance Plot')
            plt.xticks(rotation=90)
            plt.xticks(range(len(column_names)), column_names)
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)
            buffer.seek(0)
            img = buffer.read()
            buffer.close()
            story.append(Spacer(1, 12))
            story.append(Paragraph("Feature Importance Plot:", styles["Normal"]))
            story.append(Spacer(1, 12))
            story.append(Image(BytesIO(img), 500, 400))
            story.append(Spacer(1, 12))

            doc.build(story)


        # Generate and download report
        generate_report_RFC()

        ## Download generated report
        st.download_button(
            label="Download Report",
            data=open("RFC_model_report.pdf", "rb").read(),
            file_name="RFC_model_report.pdf",
            mime='application/pdf',
            key="button_report_RFC"
        )


        # Download excel file
        excel_buffer = BytesIO()
        excel_files = download_excel_files(excel_buffer, df, without_NA_RFC, all_df_RFC, train_set_RFC, valid_set_RFC, train_scaled_RFC_df, valid_scaled_RFC_df)
        st.download_button(
            label='Download Excel',
            data=excel_buffer,
            file_name='RFC_excel.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key="button_excel_RFC"
        )


# Neural Network (NN) function
def nn_tab():

    st.title("Neural Network (NN)")
    st.markdown('##### Make a predictive model using the NN method')

    # Choose descriptors to model + observed value/effect
    st.subheader('Create your dataset to modeling')

    # X variables
    X_NN = st.multiselect('Select descriptors to build the model:', df.columns, key="x_multiselect_NN")

    # y variable
    y_NN = st.multiselect('Select the property you would like to model:', df.columns, key="y_multiselect_NN")

    # Add a flag to control the flow of execution
    execute_nn = True

    # Combine selected variables
    df_NN = create_dataset(df, X_NN, y_NN)
    st.markdown('##### The dataset created by you:')
    st.write(df_NN)


    # Removing NA values
    without_NA_NN = df_NN.dropna(axis=0, how="any")
    st.markdown(" ")
    st.markdown('##### Your dataset after removing NA values looks like this:')

    if st.button('View dataset', key='button_NA_NN'):
        st.write(without_NA_NN)
    else:
        st.write(' ')

    # Usuwamy wybrane kolumny 'X' i 'y' z całego zbioru danych 'without_NA'
    without_NA_X_NN = without_NA_NN.drop(columns=y_NN)
    without_NA_y_NN = without_NA_NN[y_NN]


    # Categorical variables
    st.subheader('Categorical variables converting')
    cat_variables_NN = st.multiselect('Choose categorical variable(s) to separate descriptors:', without_NA_X_NN.columns, key="cat_var_NN")

    if cat_variables_NN:
        X_df_2_NN = separate_categorical(without_NA_X_NN, cat_variables_NN)
        st.markdown(" ")
        st.markdown('##### Your dataset with categorical variables separated looks like this:')
        all_df_NN = X_df_2_NN.join(without_NA_y_NN)
        st.write(all_df_NN)
    else:
        st.markdown('##### You have not selected any categorical variable to separate. Your dataset remains unchanged:')
        X_df_2_NN = without_NA_X_NN
        all_df_NN = X_df_2_NN.join(without_NA_y_NN)
        st.write(all_df_NN)


    # Data splitting (train and test sets)
    st.subheader("It's time to split your data for traning and validation sets!")
    # Choose a method & split data

    col1, col2 = st.columns(2)

    with col1:
        method_NN = st.radio("###### Which method do you choose?", 
                                ("Scikit-learn (random)", "Kennard Stone"), key="method_split_NN")
    
    with col2:
        split_NN = (st.slider("###### How much of the harvest will you devote to testing the model? [%]", min_value=10, max_value=40, key="split_NN"))/100


    # Splitting
    if method_NN == "Scikit-learn (random)":
        X_train_NN, X_test_NN, y_train_NN, y_test_NN = split_data(X_df_2_NN, without_NA_y_NN, method_NN, split_NN)
    else:
        X_train_NN, X_test_NN, y_train_NN, y_test_NN = split_data(X_df_2_NN, without_NA_y_NN, method_NN, split_NN)

    st.markdown(' ')
    st.subheader("View your training and validation sets")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Training Set")
        train_set_NN = X_train_NN.join(y_train_NN)
        if st.button('View Training Set', key="button_train_set_NN"):
            st.write(train_set_NN)
        else:
            st.write(' ')
    with col2:
        st.markdown("##### Validation Set")
        valid_set_NN = X_test_NN.join(y_test_NN)
        if st.button('View Validation Set', key="button_valid_set_NN"):
            st.write(valid_set_NN)
        else:
            st.write(' ')

    # Instead of error, show messages below   
    if not X_NN:
        st.warning("Please select descriptors to build the model.")
        execute_nn = False
    elif not y_NN:
        st.warning("Please select the property you'd like to model.")
        execute_nn = False
    elif not all(X_df_2_NN.dtypes.apply(lambda x: np.issubdtype(x, np.number) or np.issubdtype(x, np.bool_))):
        st.warning("Ensure all selected descriptors are either in numeric or boolean format.")
        execute_nn = False

    if execute_nn:      
        # Standardization
        st.subheader("Standardize your data")
        st.markdown("Standardize the data for better model performance.")
        X_train_NN_std, X_test_NN_std = standardize_data(X_train_NN, X_test_NN)

        st.markdown("##### Training Set (Standardized)")
        train_scaled_NN_df = pd.DataFrame(X_train_NN_std)
        st.write(train_scaled_NN_df)

        st.markdown("##### Validation Set (Standardized)")
        valid_scaled_NN_df = pd.DataFrame(X_test_NN_std)
        st.write(valid_scaled_NN_df)


        # Create a model!
        st.subheader("Create a model and choose parameters to build it!")

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            units = st.number_input('Insert a number of units:', 5, 50, key="units_NN")
            activation = st.radio('Choose an activation method:', ('relu', 'tanh', 'softmax'), key="activation_NN")
            kernel_initializer = st.radio('Choose a kernel initializer:', ('random_uniform', 'normal'), key="kernel_initializer_NN")

        with col2:
            loss = st.radio('Choose a loss function:', ('binary_crossentropy', 'hinge'), key="loss_NN")
            optimizer = st.radio('Choose a optimizer:', ('adam', 'sgd'), key="optimizer_NN")


        def create_model(units, activation, kernel_initializer, loss, optimizer):

            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(X_train_NN_std.shape[1],)),
                keras.layers.Dense(units=units, activation=activation,kernel_initializer=kernel_initializer),
                keras.layers.Dense(units=units, activation=activation,kernel_initializer=kernel_initializer),
                keras.layers.Dense(units=units, activation=activation,kernel_initializer=kernel_initializer),
                keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='uniform')

            ])
        
            # compile the keras model
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

            return model
        
        
        # Select model hyperparameters (GridSearchCV or mannual)
        st.subheader("It's time to choose hyperparameter your model!")

        hyper_method_NN = st.radio("###### How would you like to select hyperparameters?", 
                                ("GridSearchCV", "Yourself"), key="hyperparm_NN")

        # GridSearchCV
        if hyper_method_NN == "GridSearchCV":
            param_grid_NN = {'batch_size': [50, 100, 200],
                            'epochs': [1, 10, 20]}
            
            # Create the KerasClassifier object with the create_model function
            model = KerasClassifier(build_fn=create_model, units=units, activation=activation,
                                    kernel_initializer=kernel_initializer, loss=loss, optimizer=optimizer)
            grid_search_NN = GridSearchCV(estimator=model, param_grid=param_grid_NN, cv=10, verbose=True)
            grid_search_NN.fit(X_train_NN_std, y_train_NN)
            st.write(f"Best score: {grid_search_NN.best_score_:.2f} using {grid_search_NN.best_params_}")

            # DTC model by gridsearchCV
            model.fit(X_train_NN_std, y_train_NN, batch_size=grid_search_NN.best_params_['batch_size'], epochs=grid_search_NN.best_params_['epochs'])


        # Mannual way
        else:
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                batch_size = st.number_input('Insert a batch size:', 5, 50, key="batch_size_NN")

            with col2:
                epochs = st.number_input('Insert a number of epochs', 1, 50, key="epochs_NN")

            # Decision Tree model by user
            model = KerasClassifier(build_fn=create_model, units=units, activation=activation,
                                    kernel_initializer=kernel_initializer, loss=loss, optimizer=optimizer)
            model.fit(X_train_NN_std, y_train_NN, batch_size=batch_size, epochs=epochs)

        # Predicted values
        y_pred_NN = model.predict(X_test_NN_std)
        y_pred_train_NN = model.predict(X_train_NN_std)


        # Check the quality of the model
        st.markdown(' ')
        st.subheader("Let's check how our model is doing")
        st.markdown("Below you can find calculated statistics, allowing you to assess the performance, as well as the correctness of the created predictive model.")

        ## !CHANGE: the user can change colors of confustion matrixes itself
        ## !CHANGE: generowanie grafik oddzielnie (rozne formaty), tabele oddzielnie do .xlsx/.csv

        # Statistics
        train_stats_NN = generate_model_statistics(y_train_NN, y_pred_train_NN)
        valid_stats_NN = generate_model_statistics(y_test_NN, y_pred_NN)

        # Plot confusion matrices side by side
        # Display confusion matrices
        st.subheader("Confusion Matrices")
        cf_matrix_train, cf_matrix_valid, labels_train, labels_valid = generate_confusion_matrix_plot(y_train_NN, y_pred_train_NN, y_test_NN, y_pred_NN)

        # Plot confusion matrices side by side
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # color selecting for train confusion matrix
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            color_train = st.color_picker('Pick a color for confusion matrix (training set)', '#5F9EA0', key="color_train")

        # color selecting for valid confusion matrix
        with col2:
            color_valid = st.color_picker('Pick a color for confusion matrix (validation set)', '#8FBC8F', key="color_valid")

        cmap = sns.light_palette(color_train, as_cmap=True)
        sns.heatmap(cf_matrix_train, annot=labels_train, fmt='', cmap=cmap, ax=ax[0])
        ax[0].set_title('Confusion matrix for training set')

        cmap = sns.light_palette(color_valid, as_cmap=True)
        sns.heatmap(cf_matrix_valid, annot=labels_valid, fmt='', cmap=cmap, ax=ax[1])
        ax[1].set_title('Confusion matrix for validation set')
        st.pyplot(fig)

        # Display model statistics
        st.subheader("Model Statistics")
        col1, col2 = st.columns([0.55,0.45])

        with col1:
            st.markdown("##### Training Set")
            st.markdown(f"Accuracy: {train_stats_NN[0]:.2f}")
            st.markdown(f"Precision: {train_stats_NN[1]:.2f}")
            st.markdown(f"Recall: {train_stats_NN[2]:.2f}")
            st.markdown(f"F1 Score: {train_stats_NN[3]:.2f}")
            st.markdown(f"MCC: {train_stats_NN[4]:.2f}")

        with col2:
            st.markdown("##### Validation Set")
            st.markdown(f"Accuracy: {valid_stats_NN[0]:.2f}")
            st.markdown(f"Precision: {valid_stats_NN[1]:.2f}")
            st.markdown(f"Recall: {valid_stats_NN[2]:.2f}")
            st.markdown(f"F1 Score: {valid_stats_NN[3]:.2f}")
            st.markdown(f"MCC: {valid_stats_NN[4]:.2f}")


        # Prepare to report
        counts_NN = without_NA_y_NN.value_counts().tolist()
        column_names_list = X_df_2_NN.columns
        output_str_NN = ",\n".join(column_names_list)  # Dodaj przecinki pomiędzy nazwami deskryptorów
        split_NN_train = 1 - split_NN


        st.subheader('Download all excel files and report!')
        # Function to generate a PDF report
        def generate_report_NN():

            doc = SimpleDocTemplate("NN_model_report.pdf", pagesize=letter)
            story = []

            # Header
            styles = getSampleStyleSheet()
            story.append(Paragraph("Model Evaluation Report - NN", styles["Title"]))
            story.append(Spacer(1, 12))

            # Data
            story.append(Paragraph("Dataset", styles["Heading3"]))
            story.append(Paragraph(f"Objects: {X_df_2_NN.shape[0]}", styles["Normal"]))
            story.append(Paragraph(f"Descriptors: {X_df_2_NN.shape[1]}", styles["Normal"]))
            story.append(Paragraph(f"Classes: Negative ({counts_NN[0]}), Positive ({counts_NN[1]})", styles["Normal"]))
            story.append(Paragraph(f"Descriptors - names: {output_str_NN}", styles["Normal"]))
            story.append(Spacer(1, 12))
            
            # Split data
            story.append(Paragraph("Data split", styles["Heading3"]))
            story.append(Paragraph(f"Method: {method_NN}", styles["Normal"]))
            if hyper_method_NN == "GridSearchCV":
                story.append(Paragraph(f"Best hyperparameters: {grid_search_NN.best_params_}", styles["Normal"]))
            else:
                story.append(Paragraph(f"batch_size: {batch_size}", styles["Normal"]))
                story.append(Paragraph(f"epochs: {epochs}", styles["Normal"]))

            story.append(Spacer(1, 12))

            # Confusion Matrices
            story.append(Paragraph("Confusion Matrices", styles["Heading3"]))
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            cmap = sns.light_palette(color_train, as_cmap=True)
            sns.heatmap(cf_matrix_train, annot=labels_train, fmt='', cmap=cmap, ax=ax[0])
            ax[0].set_title('Confusion matrix for training set')

            cmap = sns.light_palette(color_valid, as_cmap=True)
            sns.heatmap(cf_matrix_valid, annot=labels_valid, fmt='', cmap=cmap, ax=ax[1])
            ax[1].set_title('Confusion matrix for validation set')

            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)

            buffer.seek(0)
            img = buffer.read()
            buffer.close()
            story.append(Spacer(1, 12))
            story.append(Paragraph("Confusion matrices for training and validation sets:", styles["Normal"]))
            story.append(Spacer(1, 12))
            story.append(Image(BytesIO(img), 500, 200))
            story.append(Spacer(1, 12))

            # Model Statistics
            story.append(Paragraph("Model Statistics", styles["Heading3"]))
            data = [
                ["Statistic", "Training Set", "Validation Set"],
                ["Accuracy", f"{train_stats_NN[0]:.2f}", f"{valid_stats_NN[0]:.2f}"],
                ["Precision", f"{train_stats_NN[1]:.2f}", f"{valid_stats_NN[1]:.2f}"],
                ["Recall", f"{train_stats_NN[2]:.2f}", f"{valid_stats_NN[2]:.2f}"],
                ["F1 Score", f"{train_stats_NN[3]:.2f}", f"{valid_stats_NN[3]:.2f}"],
                ["MCC", f"{train_stats_NN[4]:.2f}", f"{valid_stats_NN[4]:.2f}"]
            ]

            t = Table(data)
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), '#8a8a8a'),
                                ('TEXTCOLOR', (0, 0), (-1, 0), '#ffffff'),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), '#f7f7f7'),
                                ('GRID', (0, 0), (-1, -1), 1, '#a0a0a0'),
                                ]))
            story.append(t)
            story.append(Spacer(1, 12))

            doc.build(story)


        # Generate and download report
        generate_report_NN()

        ## Download generated report
        st.download_button(
            label="Download Report",
            data=open("NN_model_report.pdf", "rb").read(),
            file_name="NN_model_report.pdf",
            mime='application/pdf',
            key="button_report_NN"
        )


        # Download excel file
        excel_buffer = BytesIO()
        excel_files = download_excel_files(excel_buffer, df, without_NA_NN, all_df_NN, train_set_NN, valid_set_NN, train_scaled_NN_df, valid_scaled_NN_df)
        st.download_button(
            label='Download Excel',
            data=excel_buffer,
            file_name='NN_excel.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key="button_excel_NN"
        )



## MAIN FUNCTION

st.sidebar.header("Choose your goal")
tabs = ["📊 Data Preprocessing", "🔎 Principal Component Analysis (PCA)", "🏘️ K-Nearest Neighbour (KNN)", "🎰 Support Vector Machine (SVM)", "🌱 Decision Tree", "🌳 Random Forest", "🕸️ Neural Network"]
active_tab = st.sidebar.selectbox("Select a tab:", tabs)  # Let the user choose the active tab

st.sidebar.markdown("----")
st.sidebar.header("Data transfer")
uploaded_file = st.sidebar.file_uploader("Upload your file here...", type=['xlsx'])


if uploaded_file is not None:

    file_details = {
        "Filename":uploaded_file.name,
        "FileType":uploaded_file.type,
        "FileSize":uploaded_file.size
        }

    wb = openpyxl.load_workbook(uploaded_file)

    ## Show Excel file
    st.sidebar.subheader("File details:")
    st.sidebar.json(file_details,expanded=False)
    st.sidebar.markdown("----")

    ## Select sheet
    sheet_selector = st.sidebar.selectbox("Select sheet:",wb.sheetnames)     
    df = pd.read_excel(uploaded_file,sheet_selector)

    # Checking the active tab and displaying content
    if active_tab == "📊 Data Preprocessing":
        data_preprocessing_tab()

    elif active_tab == "🔎 Principal Component Analysis (PCA)":
        pca_tab()

    elif active_tab == "🏘️ K-Nearest Neighbour (KNN)":
        knn_tab()

    elif active_tab == "🎰 Support Vector Machine (SVM)":
        svm_tab()

    elif active_tab == "🌱 Decision Tree":
        dtc_tab()

    elif active_tab == "🌳 Random Forest":
        rfc_tab()
    
    elif active_tab == "🕸️ Neural Network":
        nn_tab()


#------ ELSE ------#
else:

    if active_tab == "📊 Data Preprocessing":
        st.title('Data preprocessing')
        st.markdown('##### Explore the dataset chosen by you and check how it presents')
        st.markdown("Submit your data in the left section to see what the data looks like!")

    elif active_tab == "🔎 Principal Component Analysis (PCA)":
        st.title("Principal Components Analysis (PCA)")
        st.markdown('##### You can use the PCA method to observer the relationship between variables and objects in your dataset. In addition, it is helpful in selecting descriptors for predictive models')
        st.markdown("Submit your data in the left section to see what the data looks like!")

    elif active_tab == "🏘️ K-Nearest Neighbour (KNN)":
        st.title("K-Nearest Neighbour method (KNN)")
        st.markdown('##### Make a predictive model using the k-NN method')
        st.markdown("Submit your data in the left section to see what the data looks like!")

    elif active_tab == "🎰 Support Vector Machine (SVM)":
        st.title("Support Vector Machine (SVM)")
        st.markdown('##### Make a predictive model using the SVM method')
        st.markdown("Submit your data in the left section to see what the data looks like!")

    elif active_tab == "🌱 Decision Tree":
        st.title("Decision Tree Classifier (DTC)")
        st.markdown('##### Make a predictive model using the DTC method')
        st.markdown("Submit your data in the left section to see what the data looks like!")

    elif active_tab == "🌳 Random Forest":
        st.title("Random Forest Classifier (RFC)")
        st.markdown('##### Make a predictive model using the RFC method')
        st.markdown("Submit your data in the left section to see what the data looks like!")

    else:
        st.title("🕸️ Neural Network (NN)")
        st.markdown('##### Make a predictive model using the NN method')
        st.markdown("Submit your data in the left section to see what the data looks like!")

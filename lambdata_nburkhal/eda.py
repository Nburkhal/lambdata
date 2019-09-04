import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.multiclass import unique_labels
from pdpbox.pdp import pdp_interact, pdp_interact_plot
import shap


# Get descriptive statistics for numeric and non-numeric columns of a dataframe
def descriptive(df):
    """ Get descriptive statistics from a pandas dataframe.

    return objects should be set to 2 different variables, i.e.:

    num, cat = descriptive(df)

    Returns descriptive statistics for numeric and non-numeric data.
    """

    return df.describe(), df.describe(exclude='number')


# Plot a confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    """Plots a confusion matrix for a ML model.

    y_true = target y vector 
    y_pred = ML model's predictions 

    Model should be built and run before calling this function!

    Returns a stylized confision matrix heatmap.
    """
    # Get label names from y target vector
    labels = unique_labels(y_true)
    
    # Set column labels for confusion matrix
    columns = [f'Predicted {label}' for label in labels]
    # Set index labels for confusion matrix
    index = [f'Actual {label}' for label in labels]
    
    # Create pandas dataframe to store info prior to plotting
    table = pd.DataFrame(confusion_matrix(y_true, y_pred),
                         columns=columns,
                         index=index)
    
    return sns.heatmap(table, annot=True, fmt='d', cmap='YlGnBu')


# Define function that pulls encoder mappings for the feature we specify
def feature_mapping(feature, encoder):
    """Assign variable names to encoder's ordering.

    Used for graphing purposes.

    feature = the values you want to extract
    encoder = the encoder you used on your data (ce.OrdinalEncoder(), 
              ce.OneHotEncoder(), etc.)

    Set function's outputs to 3 separate variables to get the values, i.e.:

    feat, names, codes = feature_mapping(feature)

    Returns the feature_map, category_names, and category_codes of an 
    encoder's mapping.
    """
    for item in encoder.mapping:
        if item['col'] == feature:
            feature_mapping = item['mapping']
            
    feature_map = feature_mapping[feature_mapping.index.dropna()]
    category_names = feature_mapping.index.tolist()
    category_codes = feature_mapping.values.tolist()
    
    return (feature_map, category_names, category_codes)


# Define function that takes any 2 features and returns an interactive PDP
def interactive_plot(a, b, model, X_encoded):
    """ Takes any 2 features in a dataset and creates an interactive 
        partial dependency plot.

        Utilizes the feature_mapping function to get the values of the 
        features inputted.

        a = first feature (column name)
        b = second feature (column name)
        model = Machine Learning model (do *not* use a pipeline here)
        X_encoded = the encoded X feature dataframe (validation or test) - ensure you
                    fit_transform your training data and transform your validation/
                    test data before passing

        Returns an interactive pdp plot.
    """
    # Assign inputs as features
    features = [a, b]   
    

    # Build interaction model
    interaction = pdp_interact(
        model=model,
        dataset=X_encoded, 
        model_features=X_encoded.columns, 
        features=features
    )

    # Create pivot table
    pdp = interaction.pdp.pivot_table(
        values='preds',
        columns=features[0],
        index=features[1]
    )

    # Get names and codes from encoder.mapping
    _, a_names, a_codes = feature_mapping(features[0])
    _, b_names, b_codes = feature_mapping(features[1])

    # Add column & index names to pivot table
    pdp = pdp.rename(index=dict(zip(b_codes, b_names)),
                     columns=dict(zip(a_codes, a_names)))


    # Set plot's figure size
    plt.figure(figsize=(10,8))
    return sns.heatmap(pdp, annot=True, fmt='.2f', cmap='YlGnBu');


# Define explainer function to show shapley values for our test data
def explainer(row_number, positive_class, positive_class_index=1, 
              X_test, encoder, model):
    """Show individual model predictions with shapley value interpretation.
        For binary classification only!

    row_number = the row you want to analyze
    positive_class = the model's positive class
    positive_class_index = which index to set the positive class to (can be 0 or 1)
    X_test = the dataframe you want to analyze
    encoder = the encoder used to encode categorical data
    model = Machine Learning model (ensure it is *outside* of a pipeline)

    Returns a shap plot.
    """
    
    positive_class = positive_class
    positive_class_index = positive_class_index
    
    # Get & process the data for the row
    row = X_test.loc[[row_number]]
    row_processed = encoder.transform(row)
    
     # Call model for prediction
    pred = model.predict(row_processed)
    predict = pred[0]
    
    # Get predicted probability
    pred_proba = model.predict_proba(row_processed)[0,positive_class_index]
    probability = pred_proba * 100
    if pred != positive_class:
        probability = 100 - probability
    
    # Get SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_processed)
    
    shap.initjs()
    return shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values,
        features=row
    )
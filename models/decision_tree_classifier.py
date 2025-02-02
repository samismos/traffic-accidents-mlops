import pandas as pd
import numpy as np
import mlrun
import cloudpickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# Calculate metrics for each class using sklearn's built-in functions
def calculate_metrics(conf_matrix):

    if conf_matrix.ndim != 2:
        raise ValueError("Confusion matrix must be a 2D array.")
    
    # Extract true positives, false positives, false negatives
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def entrypoint(context: mlrun.MLClientCtx, train_val_data_uri, version='latest', **args):
    # Retrieve arguments with defaults
    algorithm = args.get('algorithm')

    # Convert the concatenated string back to a DataItem
    df = train_val_data_uri.as_df()

    X = df.drop(columns=['Accident_Severity'])
    y = df['Accident_Severity']
    # Second split: Train vs Validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    model = DecisionTreeClassifier()

    # Fit model
    model.fit(X_train, y_train)


    ################ VALIDATION ################
    # Validate the model on the validation set
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    # Calculate additional metrics
    val_precision = precision_score(y_val, val_predictions, average='weighted')
    val_recall = recall_score(y_val, val_predictions, average='weighted')
    val_f1 = f1_score(y_val, val_predictions, average='weighted')

    # Confusion matrix
    all_classes = np.unique(y)
    conf_matrix = confusion_matrix(y_val, val_predictions, labels=all_classes)

    # Call the function with your confusion matrix
    precision, recall, f1 = calculate_metrics(conf_matrix)
    class_mapping = {
        0: "Fatal",
        1: "Serious",
        2: "Slight"
    }

    # Log class-specific metrics
    class_f1_scores = {}  # Dictionary to store F1 scores for each class
    for i, class_name in enumerate(all_classes):
        class_f1_scores[f"{class_mapping[class_name]} - F1 Score"] = f1[i]
        context.log_results({
            f"{class_mapping[class_name]} - F1 Score": f1[i],
        })

    context.log_results({
        "Validation F1 Score": val_f1,
        }
    )
   
    # Log the model
    context.log_model(
        key=algorithm, 
        body=cloudpickle.dumps(model),
        model_file="model.pkl",
        tag=version,
        algorithm=algorithm,
        metrics={
            "f1_score": val_f1,
            **class_f1_scores
        }
    )

    # Convert confusion matrix to DataFrame for better readability
    cm_df = pd.DataFrame(conf_matrix, index=all_classes, columns=all_classes)

    # Log confusion matrix as a CSV artifact
    cm_df.to_csv('/tmp/confusion_matrix.csv')
    context.log_artifact(
        'confusion_matrix',
        local_path='/tmp/confusion_matrix.csv',
        labels={'algorithm':algorithm}
    )
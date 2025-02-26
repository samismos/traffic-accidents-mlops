import pandas as pd
import numpy as np
import cloudpickle
from pickle import load
import mlrun
from mlrun.artifacts import get_model, update_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def entrypoint(context: mlrun.MLClientCtx, model_uri, test_data_uri, version='latest', **args):
    """Evaluate a trained model on a test dataset and log results in MLRun."""
    test_df = test_data_uri.as_df()

    test_df = test_df.dropna()
    
    model_file, model_obj, _ = get_model(model_uri)
    model = load(open(model_file, "rb"))

    # Separate features and target
    X_test = test_df.drop(columns=['Accident_Severity'])
    y_test = test_df['Accident_Severity']
    
    # Ensure test data is not empty
    if X_test is None or y_test is None:
        raise ValueError("Test data (X_test, y_test) is required but missing.")

    # Predict on test data
    test_predictions = model.predict(X_test)

    # Compute classification metrics
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions, average='weighted')
    test_recall = recall_score(y_test, test_predictions, average='weighted')
    test_f1 = f1_score(y_test, test_predictions, average='weighted')

    # Compute confusion matrix
    all_classes = np.unique(y_test)
    conf_matrix = confusion_matrix(y_test, test_predictions, labels=all_classes)

    # Define class labels
    class_mapping = {0: "Fatal", 1: "Serious", 2: "Slight"}
    
    # Compute per-class F1 scores
    f1_scores_per_class = f1_score(y_test, test_predictions, average=None)
    
    # Log results in MLRun
    metrics = {
        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1 Score": test_f1
    }

    # Add per-class F1 scores to logs
    for i, class_label in enumerate(all_classes):
        metrics[f"{class_mapping[class_label]} - F1 Score"] = f1_scores_per_class[i]

    context.log_results(metrics)

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(conf_matrix, index=all_classes, columns=all_classes)
    cm_path = "/tmp/confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    
    # Log confusion matrix as artifact
    context.log_artifact("confusion_matrix", local_path=cm_path)

    # Log the trained model
    algorithm = args.get("algorithm", "model")
    context.log_model(
        key=algorithm,
        body=cloudpickle.dumps(model),
        model_file="model.pkl",
        tag=version,
        algorithm=algorithm,
        metrics=metrics
    )

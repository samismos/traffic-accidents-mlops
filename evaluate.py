import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

def entrypoint(context: mlrun.MLClientCtx, model, test_set, version='latest', **args):

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Accuracy for serious and fatal accidents
    serious_fatal_classes = [0, 1]

    # Filter out predictions and ground truth for serious and fatal accidents
    y_test_filtered = [y for y, label in zip(y_test, y_pred) if label in serious_fatal_classes]
    y_pred_filtered = [y for y in y_pred if y in serious_fatal_classes]

    # Calculate accuracy, recall, and F1 for serious and fatal accidents
    accuracy_serious_fatal = accuracy_score(y_test_filtered, y_pred_filtered)
    recall_serious_fatal = recall_score(y_test_filtered, y_pred_filtered, average='weighted')
    f1_serious_fatal = f1_score(y_test_filtered, y_pred_filtered, average='weighted')

    # Log results manually
    context.log_results({
        "accuracy_serious_fatal": accuracy_serious_fatal,
        "recall_serious_fatal": recall_serious_fatal,
        "f1_score_serious_fatal": f1_serious_fatal
    })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    all_classes = np.unique(y)
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)
    
    # Convert confusion matrix to DataFrame for better readability
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)

    # Log confusion matrix as a CSV artifact
    cm_df.to_csv('/tmp/confusion_matrix.csv')
    context.log_artifact(
        'confusion_matrix',
        local_path='/tmp/confusion_matrix.csv',
        labels={'algorithm':algorithm}
    )

    fig, ax = plt.subplots(figsize=(10, 8))  # You can adjust the size as needed
    xgb.plot_importance(model, importance_type='weight', ax=ax)
    plt.title('Feature Importance')
    plt.xlabel('Importance (weight)')
    
    # Save the plot to a file
    plot_filename = '/tmp/feature_importance.png'
    plt.savefig(plot_filename)  # Saves the plot as an image file

    # Log the plot as an artifact in MLRun
    context.log_artifact(
        "feature_importance",  # Artifact name
        local_path=plot_filename,  # Path to the saved plot
        labels={"framework": algorithm}
    )
    plt.close()  # Close the plot to free memory
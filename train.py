import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import mlrun
import cloudpickle
import time

def entrypoint(context: mlrun.MLClientCtx, processed_dataset_uri, version='latest', **args):
    # Retrieve arguments with defaults
    model_name = args.get('ALGORITHM')

    # Convert the concatenated string back to a DataItem
    df = processed_dataset_uri.as_df()

    # Ensure data is loaded
    while df is None or df.empty:
        print("Waiting for dataset to load...")
        time.sleep(1)  # Wait for 1 second before checking again
    
    # Prepare data
    # print(df.head())
    X = df.drop(columns=['Accident_Severity'])
    y = df['Accident_Severity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    # Assuming 'args' is a dictionary that contains the key 'ALGORITHM'
    if args.get("ALGORITHM") == 'decision_tree_classifier':
        model = DecisionTreeClassifier()
    elif args.get("ALGORITHM") == 'random_forest_classifier':
        model = RandomForestClassifier()
    elif args.get("ALGORITHM") == 'xgboost':
        model = XGBClassifier()
    else:
        print("Invalid algorithm choice. Please choose 'decision_tree_classifier', 'random_forest_classifier', or 'xgboost'.")
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # for actual, predicted in zip(y_test, y_pred):
    #     print(f"Actual: {actual}  Predicted: {predicted}")

    # Log results manually
    context.log_results({"accuracy": accuracy})

    # Optionally, log the model as an artifact
    context.log_model(
        key=model_name, 
        body=cloudpickle.dumps(model),
        model_file="model.pkl",
        tag=version,
        algorithm='DecisionTreeClassifier',
        metrics={
            "accuracy": accuracy
        }
        )

    # You can also log other metrics, confusion matrix, etc.
    # For example, confusion matrix
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
        labels={"framework":"DecisionTreeClassifier"}
    )

    # Assuming 'y_test' and 'y_pred' are the ground truth and predictions
    serious_fatal_classes = [0, 1]

    # Filter out predictions and ground truth for serious and fatal accidents
    y_test_filtered = [y for y, label in zip(y_test, y_pred) if label in serious_fatal_classes]
    y_pred_filtered = [y for y in y_pred if y in serious_fatal_classes]

    # Calculate accuracy for serious and fatal accidents
    accuracy_serious_fatal = accuracy_score(y_test_filtered, y_pred_filtered)

    # Log results manually
    context.log_results({"accuracy_serious_fatal": accuracy_serious_fatal})
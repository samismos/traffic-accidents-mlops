import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree
import mlrun
from mlrun.frameworks.sklearn import apply_mlrun
import cloudpickle
import os
from pathlib import Path
from dotenv import load_dotenv
import importlib
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
from mlrun import get_dataitem
from sklearn.metrics import confusion_matrix

def model_server_init(project_name, model_path):

    # Initialize the MLRun project object
    project = mlrun.get_or_create_project(
        project_name, context="./"
    )

    serving_function_image = "mlrun/mlrun"
    serving_model_class_name = "mlrun.frameworks.sklearn.SklearnModelServer"

    # Create a serving function
    serving_fn = project.set_function(
        name="serving", kind="serving", image=serving_function_image
    )

    # Add a model, the model key can be anything we choose. The class will be the built-in scikit-learn model server class
    model_key = "scikit-learn"
    serving_fn.add_model(
        key=model_key, model_path=model_path, class_name=serving_model_class_name
    )

        # Test data to send
    my_data = {"inputs": [[24, 0,], [38, 1]]}

    # Create a mock server in order to test the model
    mock_server = serving_fn.to_mock_server()

    # Test the serving function
    mock_server.test(f"/v2/models/{model_key}/infer", body=my_data)

#     # Deploy the serving function
#     serving_fn.apply(mlrun.auto_mount()).deploy()

#     # Check the result using the deployed serving function
#     serving_fn.invoke(path=f"/v2/models/{model_key}/infer", body=my_data)


def entrypoint(context: mlrun.MLClientCtx, **args):
    # Retrieve arguments with defaults
    dataset_uri = args.get('DATASET')
    model_name = args.get('MODEL_NAME')
    
    # Load dataset
    dataset = mlrun.get_dataitem(dataset_uri).as_df()

    # Ensure data is loaded
    while dataset is None or dataset.empty:
        print("Waiting for dataset to load...")
        time.sleep(1)  # Wait for 1 second before checking again

    # Proceed once data is loaded
    print(f"Data is ready with shape: {dataset.shape}")
    
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Assuming 'genre' is the column containing the string labels 'Classical' and 'HipHop'
    dataset['genre'] = label_encoder.fit_transform(dataset['genre'])
    
    # Prepare data
    # print(dataset.head())
    X = dataset.drop(columns=['genre'])
    y = dataset['genre']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    # Initialize model
    model = DecisionTreeClassifier()
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log results manually
    context.log_results({"accuracy": accuracy})

    # Optionally, log the model as an artifact
    context.log_model(key=model_name, body=cloudpickle.dumps(model), model_file="model.pkl")

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
    
    # Optionally, print the confusion matrix to the logs
    print("Confusion Matrix:")
    print(cm_df)
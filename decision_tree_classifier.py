import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import mlrun
from mlrun import get_dataitem
import cloudpickle
import time

def entrypoint(context: mlrun.MLClientCtx, **args):
    # Retrieve arguments with defaults
    dataset_uri = args.get('DATASET_URI')
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=110)
    
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
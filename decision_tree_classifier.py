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
    dataset = args.get('DATASET')
    model_name = args.get('MODEL_NAME')
    
    # Load dataset
    music_data = mlrun.get_dataitem(dataset).as_df()
    
    # Prepare data
    X = music_data.drop(columns=['genre'])
    y = music_data['genre']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize and fit model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    # Log model
    context.log_model(key=model_name, body=cloudpickle.dumps(model), model_file="model.pkl")
    
    # Apply MLRun (assuming apply_mlrun is defined elsewhere)
    apply_mlrun(model=model, model_name=model_name, x_test=X_test, y_test=y_test)



    # model_store_path = f"store://models/{project_name}/{function_name}-{handler}_{project_name}#{model_tag}"
    # print(model_store_path)
    # model_server_init(project_name, model_store_path)
    
    # print(model_store_path)
    
    # decision-tree-classifier.serve_model();
#     predictions = model.predict(X_test.values)

#     score = accuracy_score(y_test, predictions)
#     print(score)
#     print(predictions)
   
import pandas as pd
import numpy as np
import mlrun
from mlrun.artifacts import get_model, update_model
import cloudpickle
from pickle import load
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def entrypoint(context: mlrun.MLClientCtx, train_data_uri, version='latest', **args):
    # Retrieve arguments with defaults
    algorithm = args.get('algorithm')
    model_uri = args.get('model_uri')

    # Convert URI to dataFrame
    df = train_data_uri.as_df()

    X = df.drop(columns=['Accident_Severity'])
    y = df['Accident_Severity']
    # Train vs Test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights = np.array([class_weights[class_] for class_ in y_train])

    # Load existing model if provided, else initialize a new model
    if model_uri:
        print(f"Loading existing model from {model_uri}")
        model_file, model_obj, extra_data = get_model(model_uri)
        model = load(open(model_file, "rb"))
    else:
        print("model_uri not found... Training new model.")
        # Initialize model
        model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='multi:softmax',  # Change to multi-class classification
                num_class=3,                # Number of classes in your target variable (3 for Slight, Serious, Fatal)
                random_state=42,
                eval_metric='mlogloss' ,     # Use mlogloss for multi-class classification
            )

    # Fit model
    model.fit(X_train, y_train, sample_weight=weights)

    context.log_model(
        key=algorithm, 
        body=cloudpickle.dumps(model),
        model_file="model.pkl",
        tag=version,
        algorithm=algorithm,
        )

    # Save and log test data for evaluation
    test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    test_data_path = '/tmp/test_data.csv'
    test_data.to_csv(test_data_path, index=False)

    # Log test data as an artifact
    context.log_artifact(
        'test_data',
        local_path=test_data_path,
        tag=version,
        labels={'rows': len(test_data)}
    )
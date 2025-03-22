# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlrun
from mlrun.artifacts import get_model, update_model
import cloudpickle
from pickle import load
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier



def train_then_evaluate(model, X_train, y_train, X_test, y_test, should_train=False):
    if should_train:
        print("Training model...\n")
        model.fit(X_train, y_train)
    else:
        print("Training was skipped...\n")
    """Evaluates the model and prints performance metrics."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    # Round all values in classification report to 4 decimal places
    report_dict = classification_report(y_test, preds, output_dict=True)
    for key in report_dict:
        if isinstance(report_dict[key], dict):
            report_dict[key] = {metric: round(value, 4) for metric, value in report_dict[key].items()}
        else:
            report_dict[key] = round(report_dict[key], 4)

    report_df = pd.DataFrame(report_dict).transpose()
    print("Classification Report (0 - Fatal, 1 - Serious, 2 - Slight) :")
    print(report_df)
    cm = confusion_matrix(y_test, preds)
    
    return model, acc, cm, report_df

def entrypoint(context: mlrun.MLClientCtx, X_TRAIN_URI, Y_TRAIN_URI, X_TEST_URI, Y_TEST_URI, **args):

    X_train = X_TRAIN_URI.as_df()
    X_test = X_TEST_URI.as_df()
    Y_train = Y_TRAIN_URI.as_df()
    Y_test = Y_TEST_URI.as_df()

    # Define models
    models = {
        "decision_tree_classifier": DecisionTreeClassifier(),
        "random_forest_classifier": RandomForestClassifier(),
        "xg_boost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='multi:softmax',  # Change to multi-class classification
            num_class=3,                # Number of classes in your target variable (3 for Slight, Serious, Fatal)
            random_state=42,
            eval_metric='mlogloss' ,     # Use mlogloss for multi-class classification
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=200,
            random_state=42,
        )
    }
    model = []
    model_name = args.get("ALGORITHM")
    model_uri = args.get("MODEL_URI")

    if model_uri:
        print(f"Loading existing model from {model_uri}")
        model_file, model_obj, _ = get_model(model_uri)
        model = load(open(model_file, "rb"))
        if not model_file:
            raise FileNotFoundError(f"Model file {model_file} does not exist. Check model_uri: {model_uri}")
    else:
        if not model_name:
            raise ValueError("Missing 'ALGORITHM' argument. Please specify a valid model name.")
        if model_name not in models:
            raise ValueError(f"Invalid model name '{model_name}'. Choose from {list(models.keys())}.")
        print(f"No model_uri given, a new {model_name} model will be trained.")
        model = models[model_name]
    
    print("\n============================")
    print(f"Model: {model_name}")
    print("============================")

    model, acc, cm, report = train_then_evaluate(model, X_train, Y_train, X_test, Y_test, args.get("SHOULD_TRAIN"))

    # Log confusion matrix
    cm_path = "/tmp/confusion_matrix.csv"
    pd.DataFrame(cm).to_csv(cm_path, index=False)
    context.log_artifact("confusion_matrix", local_path=cm_path, artifact_path=context.artifact_subpath("cm"))

    # Log classification report
    report_path_csv = "/tmp/classification_report.csv"
    report.to_csv(report_path_csv)

    context.log_artifact("classification_report_csv", local_path=report_path_csv, artifact_path=context.artifact_subpath("report"))

    metrics = {
        "Test Accuracy": acc,
    }

    context.log_model(
        key=model_name, 
        body=cloudpickle.dumps(model),
        model_file="model.pkl",
        tag=args.get("VERSION"),
        algorithm=model_name,
        metrics=metrics
        )
    context.log_results(metrics)
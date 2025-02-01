import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import mlrun
import cloudpickle
import time
import matplotlib.pyplot as plt
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

def entrypoint(context: mlrun.MLClientCtx, processed_dataset_uri, version='latest', **args):
    # Retrieve arguments with defaults
    algorithm = args.get('algorithm')

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
    if algorithm == 'decision_tree_classifier':
        model = DecisionTreeClassifier()

    elif algorithm == 'random_forest_classifier':
        model = RandomForestClassifier(
             n_estimators=100,
             class_weight='balanced',
             random_state=42
        )

    elif algorithm == 'xgboost':
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='multi:softmax',  # Change to multi-class classification
            num_class=3,                # Number of classes in your target variable (3 for Slight, Serious, Fatal)
            random_state=42,
            eval_metric='mlogloss'      # Use mlogloss for multi-class classification
        )

    else:
        print("Invalid algorithm choice. Please choose 'decision_tree_classifier', 'random_forest_classifier', 'xgboost' or 'lightgbm.")
    
    # Fit model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log results manually
    context.log_results({"accuracy": accuracy})

    # Optionally, log the model
    context.log_model(
        key=algorithm, 
        body=cloudpickle.dumps(model),
        model_file="model.pkl",
        tag=version,
        algorithm=algorithm,
        metrics={
            "accuracy": accuracy,
            "recall": recall,
            "f1_score": f1
        }
        )

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

    # # Create an explainer
    # explainer = ClassifierExplainer(model, X_test, y_test)
    # print(X.dtypes)

    # # Start the ExplainerDashboard
    # dashboard = ExplainerDashboard(explainer, shap_interaction=False)
    # # dashboard.run()

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
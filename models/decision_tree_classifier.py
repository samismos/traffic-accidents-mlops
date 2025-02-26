import pandas as pd
import mlrun
import cloudpickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def entrypoint(context: mlrun.MLClientCtx, train_data_uri, version='latest', **args):
    # Retrieve arguments with defaults
    algorithm = args.get('algorithm')

    # Convert the concatenated string back to a DataItem
    df = train_data_uri.as_df()

    X = df.drop(columns=['Accident_Severity'])
    y = df['Accident_Severity']
    # Second split: Train vs Validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    model = DecisionTreeClassifier()

    # Fit model
    model.fit(X_train, y_train)

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
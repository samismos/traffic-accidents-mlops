import mlrun
import os
from dotenv import load_dotenv
from datetime import datetime

def entrypoint(context: mlrun.MLClientCtx, **params):

    # Create or get the MLRun project
    project = mlrun.get_or_create_project(name=params.get('PROJECT_NAME'), context="./")

    # Timestamped versioning
    version = params.get("VERSION")

    # Makes ALGORITHM input case-insensitive
    ALGORITHM = params.get("ALGORITHM").lower()

    # Ingest the raw data set
    project.run_function(
        'ingest',
        name='ingest',
        handler=params.get('HANDLER'),
        inputs={
            'DATASET_URI': params.get("RAW_DATA_URI"),
        },
        params={
            'VERSION': version
        }
    ) 

    # Train a model
    project.run_function(
        'train_then_evaluate',
        name='train_then_evaluate',
        handler=params.get('HANDLER'),
        inputs={
            'X_TRAIN_URI': params.get("X_TRAIN"),
            'X_TEST_URI': params.get("X_TEST"),
            'Y_TRAIN_URI': params.get("Y_TRAIN"),
            'Y_TEST_URI': params.get("Y_TEST"),
        },
        params={
            'ALGORITHM': ALGORITHM,
            'SHOULD_TRAIN': True,
            'VERSION': version
        },
        # auto_build=True
    )
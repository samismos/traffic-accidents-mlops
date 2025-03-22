import mlrun
import os
from dotenv import load_dotenv
from datetime import datetime

def entrypoint(context: mlrun.MLClientCtx, **params):

    # Create or get the MLRun project
    project = mlrun.get_or_create_project(name=params.get('PROJECT_NAME'), context="./")

    # Timestamped versioning
    version = 'b_' + params.get("VERSION")

    # Makes ALGORITHM input case-insensitive
    ALGORITHM = params.get("ALGORITHM").lower()

    # Ingest the batch data set
    project.run_function(
        'ingest',
        name='ingest',
        handler=params.get('HANDLER'),
        inputs={
            'DATASET_URI': params.get("BATCH_URI"),
        },
        params={
            'VERSION': version,
            'IS_BATCH': True
        }
    ) 

    # Evaluate the model on initial dataset for reference
    project.run_function(
        'train_then_evaluate',
        name='model_eval_reference',
        handler=params.get('HANDLER'),
        inputs={
            'X_TRAIN_URI': params.get("X_TRAIN"),
            'X_TEST_URI': params.get("X_TEST"),
            'Y_TRAIN_URI': params.get("Y_TRAIN"),
            'Y_TEST_URI': params.get("Y_TEST"),
        },
        params={
            'ALGORITHM': ALGORITHM,
            'SHOULD_TRAIN': False,
            'MODEL_URI': f'{params.get("MODEL_STORE")}/{params.get("PROJECT_NAME")}/train-then-evaluate_{ALGORITHM}#0:latest',
            'VERSION': version
        }
    )
    # Evaluate the model on batch data
    project.run_function(
        'train_then_evaluate',
        name='evaluate_on_batch',
        handler=params.get('HANDLER'),
        inputs={
            'X_TRAIN_URI': params.get("B_X_TRAIN"),
            'X_TEST_URI': params.get("B_X_TEST"),
            'Y_TRAIN_URI': params.get("B_Y_TRAIN"),
            'Y_TEST_URI': params.get("B_Y_TEST"),
        },
        params={
            'ALGORITHM': ALGORITHM,
            'SHOULD_TRAIN': False,
            'MODEL_URI': f'{params.get("MODEL_STORE")}/{params.get("PROJECT_NAME")}/train-then-evaluate_{ALGORITHM}#0:latest',
            'VERSION': version
        }
    )   

    # Retrain the model on the batch data
    project.run_function(
        'train_then_evaluate',
        name='retrain_on_batch',
        handler=params.get('HANDLER'),
        inputs={
            'X_TRAIN_URI': params.get("B_X_TRAIN"),
            'X_TEST_URI': params.get("B_X_TEST"),
            'Y_TRAIN_URI': params.get("B_Y_TRAIN"),
            'Y_TEST_URI': params.get("B_Y_TEST"),
        },
        params={
            'ALGORITHM': ALGORITHM,
            'SHOULD_TRAIN': True,
            'MODEL_URI': f'{params.get("MODEL_STORE")}/{params.get("PROJECT_NAME")}/train-then-evaluate_{ALGORITHM}#0:latest',
            'VERSION': version
        }
    )

    # Evaluate the model on initial dataset
    project.run_function(
        'train_then_evaluate',
        name='evaluate_final',
        handler=params.get('HANDLER'),
        inputs={
            'X_TRAIN_URI': params.get("X_TRAIN"),
            'X_TEST_URI': params.get("X_TEST"),
            'Y_TRAIN_URI': params.get("Y_TRAIN"),
            'Y_TEST_URI': params.get("Y_TEST"),
        },
        params={
            'ALGORITHM': ALGORITHM,
            'SHOULD_TRAIN': False,
            'MODEL_URI': f'{params.get("MODEL_STORE")}/{params.get("PROJECT_NAME")}/train-then-evaluate_{ALGORITHM}#0:latest',
            'VERSION': version
        }
    )
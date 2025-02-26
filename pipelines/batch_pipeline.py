import mlrun
import os
from dotenv import load_dotenv
from datetime import datetime

def entrypoint(context: mlrun.MLClientCtx, **params):

    # Create or get the MLRun project
    project = mlrun.get_or_create_project(name=params.get('PROJECT_NAME'), context="./")

    # Timestamped versioning
    version = 'b_' + params.get("VERSION")
    print("VERSION IN BATCH_PIPELINE: ",version)

    # Ingest the raw data set
    # project.run_function(
    #     'ingest',
    #     name='ingest',
    #     handler=params.get('HANDLER'),
    #     inputs={
    #         'dataset_uri': params.get("BATCH_URI"),
    #     },
    #     params={
    #         'version': version
    #     }
    # )

    # Evaluate model on new data
    project.run_function(
        'evaluate',
        name='evaluate',
        handler=params.get('HANDLER'),
        inputs={
            'model_uri': f'{params.get("MODEL_STORE")}/{params.get("PROJECT_NAME")}/train_{params.get("ALGORITHM")}#0:latest',
            'test_data_uri': params.get("BATCH_URI"),
            'version': version
        },
        params={
            'algorithm': params.get("ALGORITHM")
        },
        auto_build=True
    )

    print(f'{params.get("MODEL_STORE")}/{params.get("PROJECT_NAME")}/train_{params.get("ALGORITHM")}#0:latest')

    # Re-train the model based on the new data
    project.run_function(
        'train',
        name='train',
        handler=params.get('HANDLER'),
        inputs={
            'train_data_uri': params.get("BATCH_URI"),
            'version': version
        },
        params={
            'algorithm': params.get("ALGORITHM"),
            'model_uri': f'{params.get("MODEL_STORE")}/{params.get("PROJECT_NAME")}/train_{params.get("ALGORITHM")}#0:latest',
        },
        auto_build=True
    )

    # Evaluate the model
    project.run_function(
        'evaluate',
        name='evaluate',
        handler=params.get('HANDLER'),
        inputs={
            'test_data_uri': params.get("TEST_DATA_URI"),
            'model_uri': f'{params.get("MODEL_STORE")}/{params.get("PROJECT_NAME")}/train_{params.get("ALGORITHM")}#0:latest',
            'version': version
        },
        params={
            'algorithm': params.get("ALGORITHM")
        },
        auto_build=True
    )
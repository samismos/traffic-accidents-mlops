import mlrun
import os
from dotenv import load_dotenv
from datetime import datetime

### Select project configuration
load_dotenv(".env/main_config.env")

### Network configuration
load_dotenv(".env/network_config.env")

# Set MLRun environment
mlrun.set_environment(
    os.getenv('MLRUN_DBPATH'),
    artifact_path=os.getenv('ARTIFACT_BASE_PATH')+os.getenv('PROJECT_NAME')
)

# Create or get the MLRun project
project = mlrun.get_or_create_project(name=os.getenv('PROJECT_NAME'), context="./")

ingest = mlrun.code_to_function(
    name="ingest", 
    kind="job",
    filename="ingest.py",
    project=os.getenv('PROJECT_NAME')
)

# Convert the Python script to a function
train = mlrun.code_to_function(
    name="train", 
    kind="job",
    filename=f"models/{os.getenv('ALGORITHM')}.py",
    project=os.getenv('PROJECT_NAME'),
    image=os.getenv('TRAIN_IMAGE'),
    tag=os.getenv('ALGORITHM')
)

# Convert the Python script to a function
evaluate = mlrun.code_to_function(
    name="evaluate", 
    kind="job",
    filename="evaluate.py",
    project=os.getenv('PROJECT_NAME'),
    image=os.getenv('TRAIN_IMAGE'),
)

# Timestamped versioning
version = datetime.now().strftime('%d%m%Y_%H%M%S')

# Run functions
ingest.run(
    name='ingest',
    handler=os.getenv('HANDLER'),
    inputs={
        'dataset_uri': os.getenv("RAW_DATA_URI"),
    },
    params={
        'VERSION': version
    }
)

train.run(
    name='train',
    handler=os.getenv('HANDLER'),
    inputs={
        'train_data_uri': os.getenv("TRAIN_DATA_URI"),
        'version': version
    },
    params={
        'algorithm': os.getenv("ALGORITHM")
    },
    auto_build=True
)

evaluate.run(
    name='evaluate',
    handler=os.getenv('HANDLER'),
    inputs={
        'model_uri': f'{os.getenv("MODEL_STORE")}/{os.getenv("PROJECT_NAME")}/train_{os.getenv("ALGORITHM")}#0:latest',
        'test_data_uri': os.getenv("TEST_DATA_URI"),
        'version': version
    },
    params={
        'algorithm': os.getenv("ALGORITHM")
    },
    auto_build=True
)
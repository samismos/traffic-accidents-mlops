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
    project=os.getenv('PROJECT_NAME'),
    image=os.getenv('TRAIN_IMAGE'),
)

# Convert the Python script to a function
train_then_evaluate = mlrun.code_to_function(
    name="train_then_evaluate", 
    kind="job",
    filename="train_then_evaluate.py",
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
        'DATASET_URI': os.getenv("RAW_DATA_URI"),
    },
    params={
        'VERSION': version
    }
)

train_then_evaluate.run(
    name='train_then_evaluate',
    handler=os.getenv('HANDLER'),
    inputs={
      'X_TRAIN_URI': os.getenv("X_TRAIN"),
      'X_TEST_URI': os.getenv("X_TEST"),
      'Y_TRAIN_URI': os.getenv("Y_TRAIN"),
      'Y_TEST_URI': os.getenv("Y_TEST"),
    },
    params={
        'ALGORITHM': os.getenv("ALGORITHM"),
        'SHOULD_TRAIN': True,
        'VERSION': version
    },
    auto_build=True
)

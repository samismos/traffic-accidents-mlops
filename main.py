import mlrun
import os
from dotenv import load_dotenv
from datetime import datetime

### Select project configuration
load_dotenv("traffic_accidents.env")

### Network configuration
load_dotenv("network_config.env")

# Set MLRun environment
mlrun.set_environment(os.getenv('MLRUN_API'), artifact_path=os.getenv('ARTIFACT_BASE_PATH')+os.getenv('PROJECT_NAME'))

# Create or get the MLRun project
project = mlrun.get_or_create_project(name=os.getenv('PROJECT_NAME'), context="./")

ingest = mlrun.code_to_function(
    name="ingest", 
    kind="job",
    filename="ingest.py"
)

# Convert the Python script to a function
train = mlrun.code_to_function(
    name="train", 
    kind="job",
    filename="train.py",
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
        'processed_dataset_uri': os.getenv("PROCESSED_DATA_URI"),
        'version': version
    },
    params={
        'algorithm': os.getenv("ALGORITHM")
    },
    auto_build=True
)
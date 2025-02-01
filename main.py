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
    filename="ingest.py" ## Change to target filename / read from env
)

# Convert the Python script to a function
train = mlrun.code_to_function(
    name="train", 
    kind="job",
    filename="decision_tree_classifier.py" ## Change to target filename / read from env
)

# Pass on the env context to the function
env = {
    "PROJECT_NAME": os.getenv("PROJECT_NAME"),
    "DATASET_URI": os.getenv("DATASET_URI"),
    "HANDLER": os.getenv("HANDLER"),
    "MODEL_TAG": os.getenv("MODEL_TAG"),
    "MODEL_NAME": os.getenv("MODEL_NAME"),
}

# Timestamped versioning
version = datetime.now().strftime('%d%m%Y_%H%M%S')


# # Run the function
# function.run(handler=os.getenv('HANDLER'), params=env)  # Specify the function to run


ingest.run(
    name='ingest',
    handler=os.getenv('HANDLER'),
    inputs={
        'dataset_uri': os.getenv("DATASET_URI"),
    },
    params={
        'VERSION': version
    }
    )

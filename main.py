import mlrun
import os
from dotenv import load_dotenv


### Select project configuration
load_dotenv("music_config.env")

### Network configuration
load_dotenv("network_config.env")

# Set MLRun environment
mlrun.set_environment(os.getenv('MLRUN_API'), artifact_path=os.getenv('ARTIFACT_BASE_PATH')+os.getenv('PROJECT_NAME'))

# Create or get the MLRun project
project = mlrun.get_or_create_project(name=os.getenv('PROJECT_NAME'), context="./")

# Convert the Python script to a function
function = mlrun.code_to_function(
    name=os.getenv('FUNCTION_NAME'), 
    kind="job",
    filename="decision_tree_classifier.py" ## Change to target filename / read from env
)

# Pass on the env context to the function
env = {
    "PROJECT_NAME": os.getenv("PROJECT_NAME"),
    "DATASET": os.getenv("DATASET_URI"),
    "FUNCTION_NAME": os.getenv("FUNCTION_NAME"),
    "HANDLER": os.getenv("HANDLER"),
    "MODEL_TAG": os.getenv("MODEL_TAG"),
    "MODEL_NAME": os.getenv("MODEL_NAME"),
}

# Run the function
function.run(handler=os.getenv('HANDLER'), params=env)  # Specify the function to run

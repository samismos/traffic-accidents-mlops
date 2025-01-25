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
    filename="script.py"
)


# Run the function
function.run(name="function-test", handler=os.getenv('HANDLER'))  # Specify the function to run

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

env = {
    'PROJECT_NAME': os.getenv("PROJECT_NAME"),
    'ALGORITHM': os.getenv("ALGORITHM"),
    'HANDLER': os.getenv("HANDLER"),
    'RAW_DATA_URI': os.getenv("RAW_DATA_URI"),
    'BATCH_URI': os.getenv("BATCH_URI"),
    'TRAIN_DATA_URI': os.getenv("TRAIN_DATA_URI"),
    'TEST_DATA_URI': os.getenv("TEST_DATA_URI"),
    'TRAIN_IMAGE': os.getenv("TRAIN_IMAGE"),
    'MODEL_STORE': os.getenv("MODEL_STORE"),
    'DATA_STORE': os.getenv("DATA_STORE"),
    'ARTIFACT_STORE': os.getenv("ARTIFACT_STORE"),
    'MLRUN_DBPATH':  os.getenv('MLRUN_DBPATH'),
    'ARTIFACT_BASE_PATH': os.getenv('ARTIFACT_BASE_PATH')
}

# Now you can access values like env['PROJECT_NAME'], env['ALGORITHM'], etc.

pipeline = mlrun.code_to_function(
    name="main_pipeline", 
    kind="job",
    filename="pipelines/main_pipeline.py",
    project=os.getenv('PROJECT_NAME'),
)

batch_pipeline = mlrun.code_to_function(
    name="batch_pipeline", 
    kind="job",
    filename="pipelines/batch_pipeline.py",
    project=os.getenv('PROJECT_NAME'),
)

# Timestamped versioning
version = datetime.now().strftime('%d%m%Y_%H%M%S')

# pipeline.run(
#     name='main_pipeline',
#     handler=os.getenv('HANDLER'),
#     params={**env, 'VERSION': version}
# )

batch_pipeline.run(
    name='batch_pipeline',
    handler=os.getenv('HANDLER'),
    params={**env, 'VERSION': version}
)
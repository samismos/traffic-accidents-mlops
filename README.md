# traffic-accidents-mlops
Predictive analysis pipeline for traffic accidents in England using MLOps and MLRun.

This repository hosts the code and resources for a predictive analysis pipeline focused on traffic accidents across England. It implements an MLOps methodology with MLRun to enable continuous training and retraining of a machine learning model using live data feeds. The project aims to provide actionable insights for improving traffic safety and resource allocation.

# Build dev environment

Inside the project directory, build the image from the Dockerfile.dev:

```docker build -f Dockerfile.dev -t mlrun-dev .```

The dev environment can be built with the command below:

```docker run -it --rm --network host -v "$PWD":/project --name mlrun-dev mlrun-dev```

Note: $PWD is a Windows shorthand for current directory, modify accordingly

Inside the dev environment, you can run the python files which use the MLRun SDK.

###  IMPORTANT 
Create/update the ```network_config.env``` file, and make sure it includes:

MLRUN_API="<insert_your_api_url_here>"
ARTIFACT_BASE_PATH=<insert_your_artifact_base_path_here>
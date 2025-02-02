# traffic-accidents-mlops
Predictive analysis system for traffic accidents in England using MLRun, an open-source MLOps framework.

This repository hosts the code and resources for a predictive analysis pipeline focused on traffic accidents across England. It implements an MLOps methodology with MLRun to enable continuous training of machine learning models using a mock data feed. The project aims to provide actionable insights for improving traffic safety and resource allocation.

# Install MLRun

While MLRun provides a few ways to install its ecosystem, this project has been developed on a locally installed Kubernetes cluster (docker-desktop). Below are the official instructions to install it on Kubernetes:

https://docs.mlrun.org/en/v1.6.4/install/kubernetes.html#install-on-kubernetes


# Build dev environment

Inside the project directory, build the image from the Dockerfile.dev:

```docker build -f Dockerfile.dev -t mlrun-dev .```

The dev environment can be built with the command below:

```docker run -it --rm --network host -v "$PWD":/project --name mlrun-dev mlrun-dev```

_Note: $PWD is a Windows shorthand for current directory, modify accordingly_

Inside the dev environment, you can run the ```main.py``` file which uses the MLRun SDK.

To select an algorithm, modify the ```ALGORITHM``` variable in the ```main_config.env``` file.

## Available Algorithm Options:

| Algorithm Name | Description                             |
|----------------|-----------------------------------------|
| `decision_tree_classifier`           | Decision Tree - A simple, interpretable classifier based on a tree-like model. |
| `random_forest_classifier`           | Random Forest - An ensemble method using multiple decision trees to improve accuracy and control overfitting. |
| `xg_boost`     | XGBoost - A highly efficient and scalable gradient boosting framework.      |
| `light_gbm`    | LightGBM - A fast, distributed, high-performance gradient boosting framework optimized for large datasets.       |

Make sure the variable matches exactly the names in this table, as it is directly mapped to a file in the ```models``` folder.
# Build trainer image

The train environment can be built with the command below:

```docker build -f Dockerfile.train -t samismos/mlrun-dev:1.7.2 .```

Push image to Docker Hub repository:

```docker push samismos/mlrun-dev:1.7.2```

###  IMPORTANT 
Create/update the ```network_config.env``` file, and make sure it includes:

MLRUN_API="<insert_your_api_url_here>" 

ARTIFACT_BASE_PATH=<insert_your_artifact_base_path_here>

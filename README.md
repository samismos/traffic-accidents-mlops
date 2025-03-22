# BSc Dissertation project: MLOps system for traffic accident severity classification
A raw prototype of a classification system for traffic accidents in England using MLRun, an open-source MLOps framework.

This repository hosts the code for a classification pipeline focused on traffic accidents across England. It implements an MLOps methodology with MLRun to enable continuous training of machine learning models using a mock data feed. The project aims to provide actionable insights for improving traffic safety and resource allocation.

# Install MLRun

While MLRun provides a few ways to install its ecosystem, this project has been developed on a locally installed Kubernetes cluster (docker-desktop). Below are the official instructions to install it on Kubernetes:

https://docs.mlrun.org/en/v1.6.4/install/kubernetes.html#install-on-kubernetes

# IMPORTANT! Local network configuration
Create/update the ```.env/network_config.env``` file, and make sure it includes:

MLRUN_API="<insert_your_api_url_here>" 

ARTIFACT_BASE_PATH=<insert_your_artifact_base_path_here>

# Build dev environment

Inside the project directory, build the image from the Dockerfile.dev:

```docker build -f Dockerfile.dev -t mlrun-dev .```

The dev environment can be built with the command below:

```docker run -it --rm --network host -v "$PWD":/project --name mlrun-dev mlrun-dev```

_Note: $PWD is a Linux shorthand for current directory, modify accordingly_

Inside the dev environment, you can run the ```main.py``` file which uses the MLRun SDK.

To select an algorithm, modify the ```ALGORITHM``` variable in the ```.env/main_config.env``` file.

## Available Algorithm Options:

| Algorithm Name | Description                             |
|----------------|-----------------------------------------|
| `decision_tree_classifier`           | Decision Tree - A simple, interpretable classifier based on a tree-like model. |
| `random_forest_classifier`           | Random Forest - An ensemble method using multiple decision trees to improve accuracy and control overfitting. |
| `xg_boost`     | XGBoost - A highly efficient and scalable gradient boosting framework.      |
| `light_gbm`    | LightGBM - A fast, distributed, high-performance gradient boosting framework optimized for large datasets.       |
| `mlp` | Multi-Layer Perceptron - A class of feedforward artificial neural networks composed of multiple layers of nodes, commonly used for classification and regression tasks. |

The "ALGORITHM" variable in the ```.env/main_config.env``` file is case-insensitive, but needs to match one of the algorithms in this table, as it is mapped directly to the dictionary in ```train_then_evaluate.py```.

# Build trainer image

MLRun provides its official ```mlrun/mlrun``` image with built-in support for popular frameworks. However, if you have different requirements, e.g. ```XGBoost```, you need to build a custom image and push it to your Docker Hub repository.

I have created a sample Dockerfile.job with included support for XGBoost and LightGBM. You may add any other dependencies. The image can be built with the command below:

```docker build -f Dockerfile.train -t <DOCKER_HUB_ID>/<IMAGE_NAME>:<TAG> .```

_Note: the <DOCKER_HUB_ID>/<IMAGE_NAME> must exactly match the name of your Docker Hub repository._

Push image to Docker Hub repository:

```docker push <YOUR_DOCKER_HUB_ID>/<IMAGE_NAME>:<TAG>```

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

## Available Algorithm Options:

| Algorithm Name | Description                             |
|----------------|-----------------------------------------|
| `decision_tree_classifier`           | Decision Tree - A simple, interpretable classifier based on a tree-like model. |
| `random_forest_classifier`           | Random Forest - An ensemble method using multiple decision trees to improve accuracy and control overfitting. |
| `xg_boost`     | XGBoost - A highly efficient and scalable gradient boosting framework.      |
| `mlp` | Multi-Layer Perceptron - A class of feedforward artificial neural networks composed of multiple layers of nodes, commonly used for classification and regression tasks. |

The "ALGORITHM" variable in the ```.env/main_config.env``` file is case-insensitive, but needs to match one of the algorithms in this table, as it is mapped directly to the dictionary in ```train_then_evaluate.py```.

# Build trainer image

MLRun provides its official ```mlrun/mlrun``` image with built-in support for popular frameworks. However, if you have different requirements, e.g. ```XGBoost```, you need to build a custom image and push it to your Docker Hub repository.

I have created a sample Dockerfile.job with included support for XGBoost and LightGBM. You may add any other dependencies. The image can be built with the command below:

```docker build -f Dockerfile.train -t <DOCKER_HUB_ID>/<IMAGE_NAME>:<TAG> .```

_Note: the <DOCKER_HUB_ID>/<IMAGE_NAME> must exactly match the name of your Docker Hub repository._

Push image to Docker Hub repository:

```docker push <YOUR_DOCKER_HUB_ID>/<IMAGE_NAME>:<TAG>```

# Build and run self-hosted runner for Github Actions integration (if applicable)

If you want to use GitHub Actions for CI, but have a locally hosted cluster, you need to schedule jobs on a self-hosted runner inside your cluster. The image is ```images/Dockerfile.runner```. 

Build the image with:

```docker build -t github-runner --build-arg GITHUB_URL=<your_repo_url> --build-arg GITHUB_TOKEN=<your_github_token> .```


Start the runner with:

```docker run --rm --network host --name github-runner github-runner```
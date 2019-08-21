# AI Platform Notebooks and Deep Learning Containers


## Creating an AI Platform Notebook based on a custom container
It is recommended to build a custom container as a derivative of one of the base Deep Learning containers.

To list the current Deep Learning containers
```
gcloud container images list --repository="gcr.io/deeplearning-platform-release"
```

To provision and AI Platform Notebook based on a custom container image for CPU.
```
export INSTANCE_NAME="custom-container-notebook-cpu"
export IMAGE_URI="gcr.io/jk-demo1/sklearn-cpu:latest"
export IMAGE_FAMILY="common-container" 
export ZONE="us-west1-a"
export INSTANCE_TYPE="n1-standard-8"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project="deeplearning-platform-release" \
        --maintenance-policy=TERMINATE \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=100GB \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata="proxy-mode=project_editors,container=$IMAGE_URI"
```

To provision and AI Platform Notebook based on a custom container image for GPU.
```
export INSTANCE_NAME="custom-container-notebook-gpu"
export IMAGE_URI="gcr.io/jk-demo1/sklearn-cpu:latest"
export IMAGE_FAMILY="common-container" 
export ZONE="us-west1-a"
export INSTANCE_TYPE="n1-standard-8"
export ACCELERATOR="type=nvidia-tesla-t4,count=2"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project="deeplearning-platform-release" \
        --maintenance-policy=TERMINATE \
        --accelerator=$ACCELERATOR \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=100GB \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata="install-nvidia-driver=True,proxy-mode=project_editors,container=$IMAGE_URI"
```

To get JupyterLab URL

```
gcloud compute instances describe "${INSTANCE_NAME}" \
  --format='value[](metadata.items.proxy-url)'
```

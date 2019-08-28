# AI Platform Notebooks and Deep Learning Containers
The following instructions are for creating AI Platform Notebooks using `gcloud` command and Cloud Shell.

## Creating an AI Platform Notebook based on a pre-configured VM image.

Specific [VM images](https://cloud.google.com/deep-learning-vm/docs/images) are available to suit your choice of framework and processor. For example, to create AI Platform Notebook based on the latest Base CPU image (that includes sklearn and pandas)

```
export INSTANCE_NAME="ai-notebook-cpu"
export ZONE="us-west1-a"
export INSTANCE_TYPE="n1-standard-8"
export IMAGE="common-cpu"


gcloud compute instances create ${INSTANCE_NAME} \
      --zone=$ZONE \
      --machine-type=${INSTANCE_TYPE} \
      --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email \
      --min-cpu-platform="Intel Skylake" \
      --image-family=${IMAGE} \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=100GB \
      --boot-disk-device-name=${INSTANCE_NAME} \
      --maintenance-policy=TERMINATE \
      --metadata="proxy-user-mail=${GCP_LOGIN_NAME}"
```

To create AI Platform Notebook based on the latest Tensorflow GPU image
```
export INSTANCE_NAME="ai-notebook-gpu"
export ZONE="us-west1-a"
export INSTANCE_TYPE="n1-standard-8"
export IMAGE="tf-latest-gpu"
export ACCELERATOR="type=nvidia-tesla-p100,count=1"

gcloud compute instances create ${INSTANCE_NAME} \
      --zone=$ZONE \
      --machine-type=${INSTANCE_TYPE \
      --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email \
      --min-cpu-platform="Intel Skylake" \
      --image-family=${IMAGE} \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=100GB \
      --accelerator=type=${ACCELERATOR} \
      --boot-disk-device-name=${INSTANCE_NAME} \
      --maintenance-policy=TERMINATE \
      --metadata="proxy-user-mail=${GCP_LOGIN_NAME},install-nvidia-driver=True"
```

## Creating an AI Platform Notebook based on a custom container.
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

## Getting URL to JupyterLab

To get JupyterLab URL

```
gcloud compute instances describe "${INSTANCE_NAME}" \
  --format='value[](metadata.items.proxy-url)'
```

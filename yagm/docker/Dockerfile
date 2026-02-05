FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
LABEL maintainer="dangnh0611@gmail.com"

COPY ./Dockerfile /workspace/

ENV NO_ALBUMENTATIONS_UPDATE=1; TIMM_USE_OLD_CACHE=1

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y zsh git build-essential make cmake ffmpeg mysql-client 

# Pip dependencies
RUN pip install notebook ipywidgets
RUN pip install --upgrade hydra-core
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda120
RUN pip install yacs pyyaml fvcore packaging ninja \
    torchvision torch_geometric torch-optimi lightning transformers datasets scikit-learn scikit-spatial scipy numba \
    polars pandas openpyxl \
    seaborn matplotlib shap ydata-profiling \
    catboost \
    albumentations==1.4.16 audiomentations \
    wandb tensorboard \
    joblib mapply setproctitle pytorch-ignite exhaustive-weighted-random-sampler lap \
    'causal-conv1d>=1.4.0' mamba-ssm Cython cython_bbox xtcocotools \
    pydicom dicomsdl python-gdcm pylibjpeg \
    optuna SQLAlchemy PyMySQL
RUN pip install git+https://github.com/rwightman/pytorch-image-models.git
RUN pip install git+https://github.com/facebookresearch/detectron2.git
RUN pip install onnx
RUN pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
RUN pip install onnxoptimizer onnxsim onnxruntime-gpu tensorrt pycuda
RUN pip install flash-attn --no-build-isolation

# Apex
WORKDIR /workspace/lib/
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /workspace/lib/apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
WORKDIR /workspace

# torch2trt
WORKDIR /workspace/lib/
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
# COPY torch2trt torch2trt
WORKDIR /workspace/lib/torch2trt
RUN python3 setup.py install

# Competition specifics
# NOTHING

WORKDIR /workspace/
CMD ["/usr/bin/zsh"]

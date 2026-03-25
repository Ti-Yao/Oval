FROM tensorflow/tensorflow:2.15.0

RUN apt-get update && apt-get install -y \
    libopenexr-dev \
    libx11-6 \
    libgl1 \
    libglx-mesa0 \
    libopengl0 \
    libxrender1

RUN pip install --upgrade pip setuptools wheel \
 && pip install --ignore-installed blinker \
 && pip install --upgrade \
     tqdm pydicom h5py scikit-learn scikit-image jupyter typing_extensions \
     numpy==1.26.2 scipy json5 jsonref matplotlib matplotlib-inline nibabel openexr-python \
     openpyxl pandas Pillow pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg seaborn \
     ipykernel streamlit streamlit_image_coordinates


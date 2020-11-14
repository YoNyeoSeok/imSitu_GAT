# FROM nvidia/cuda:cuda10.0-cudnn7-devel-ubuntu16.04

# # Miniconda3
# ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# ENV PATH /opt/conda/bin:$PATH

# RUN apt-get update --fix-missing && \
#     apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
#     apt-get clean

# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     /opt/conda/bin/conda clean --all --yes
FROM yonyeoseok/conda3:cuda10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update --fix-missing && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean

RUN /opt/conda/bin/conda update -n base -c defaults conda && \
    /opt/conda/bin/conda create -yn py36torch14 python=3.6 pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch

RUN chgrp -R conda /opt/conda && \
    chmod -R 770 /opt/conda

CMD [ "/bin/bash" ]

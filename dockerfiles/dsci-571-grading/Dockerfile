# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=ubcmds/dsci-011:latest
FROM $BASE_CONTAINER
LABEL maintainer="Elijah Willie <ewillie@cmmt.ubc.ca>"

# Install nbgrader via pip
USER root
RUN pip install nbgrader==0.6.1
RUN pip install graphviz
RUN pip install -U scikit-learn
RUN pip install xgboost
RUN jupyter nbextension install --sys-prefix --py nbgrader --overwrite 
RUN jupyter nbextension enable --sys-prefix --py nbgrader
RUN jupyter serverextension enable --sys-prefix --py nbgrader

USER $NB_UID

# run $unset `env | grep proxy | cut -d= -f1` first to unset proxy
# need to increase SWAP memory usage:
# use minoconda, it has python 3x installed already
FROM continuumio/miniconda3

#speicify docker workdir
WORKDIR /home/patchgnn

#copy neccessary files from host to docker
COPY environment.yml .

# install packages from python env, the base image allows the use of `conda` directly
RUN conda env create -f environment.yml

#add customized source code
COPY --chown=docker:docker src src
COPY --chown=docker:docker notebooks notebooks

# expose ports for jupyter notebook test
EXPOSE 8888


#taken from https://medium.com/@chadlagore/conda-environments-with-docker-82cdc9d25754 -- below commented out line won't work
#ENV PATH /opt/conda/envs/patch-gnn/bin:$PATH
#RUN echo "conda activate patch-gnn" > ~/.bashrc


#taken from https://pythonspeed.com/articles/activate-conda-dockerfile/
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "patch-gnn", "/bin/bash", "-c"]

#give full path to the src directory on docker 
ENV PYTHONPATH /home/patchgnn/src/
# install patchgnn's customized package and install ipykernel
RUN python src/setup.py install 

RUN python -m ipykernel install --user  --name patch-gnn


# specify python env path, use the python path in the docker container not on the host
#ENV PATH /opt/conda/envs/patch-gnn:$PATH
#ENV PATH /Users/coxji1/anaconda/envs/patch-gnn:$PATH
#RUN conda init bash && conda activate patch-gnn


#run jupyter notebook
#CMD ["conda", "run", "--no-capture-output", "-n", "patch-gnn", "python"]
CMD ["conda", "run", "--no-capture-output", "-n", "patch-gnn",  "jupyter", "lab", "--allow-root", "--ip='*'"]




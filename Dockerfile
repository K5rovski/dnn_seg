FROM cuda:10.1-base
MAINTAINER kristijan Petrovski

# Conda Dockerfile, as debian latest
# ----------
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# ---------


# make sure the package repository is up to date and update ubuntu
RUN \
  apt-get update && \
apt-get install -y gcc g++ curl git htop man software-properties-common unzip vim locales \
 wget gfortran graphviz libgraphviz-dev graphviz-dev pkg-config  \
 mysql-server libpq-dev default-libmysqlclient-dev



# RUN locale-gen en_US.UTF-8  && \
# export LANG=en_US.UTF-8 && \
# export LANGUAGE=en_US.UTF-8 && \
# export LC_ALL=en_US.UTF-8

ENV HOME /root


ADD setup/dnn_experiment_condaenv.yml /etc/clv_env.yml
# ADD  config/pip_reqs.txt  /etc/pip_reqs.txt

#  conda config --set auto_update_conda False && \




RUN  echo "python 3.5.*  " > /opt/conda/conda-meta/pinned && \
     conda update -n base -c defaults conda && \
	conda env create -f /etc/clv_env.yml   && \
 echo "source activate clv_env" > ~/.bashrc

# RUN	/opt/conda/envs/clv_env/bin/python -m pip install -U -r /etc/pip_reqs.txt





# File transfer


COPY . /opt/app
WORKDIR /opt/app




ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]



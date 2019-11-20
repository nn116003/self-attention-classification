FROM nvidia/cuda:10.1-base
RUN apt-get update && \
    apt-get -y install curl bzip2 locales tar && \
    curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh && \
    conda update conda && \
    apt-get -qq -y remove bzip2 && \
    apt-get -qq -y autoremove && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log && \
    conda clean --all --yes && \
    locale-gen ja_JP.UTF-8 && \
    localedef -f UTF-8 -i ja_JP ja_JP && \
    pip install torch torchtext dill revtok
ENV LANG ja_JP.UTF-8 
ENV LANGUAGE jja_JP:jp 
ENV LC_ALL ja_JP.UTF-8

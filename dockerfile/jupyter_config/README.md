Jupyter Notebook Vim 配置

## DO

```
RUN PIP_INSTALL="pip install -U --no-cache-dir --retries 20 --timeout 120 \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple" && \
    APT_INSTALL="apt-get install -y --no-install-recommends" && \
    curl -sL https://deb.nodesource.com/setup_15.x | bash - && \
    $APT_INSTALL jupyter-lab nodejs && \
    $PIP_INSTALL \
        autopep8 \
        notebook==v6.4.4 \
        jupyter \
        jupyter_contrib_nbextensions \
        jupyter_nbextensions_configurator \
        jupyterlab_vim \
        && \
    jupyter contrib nbextension install --sys-prefix && \
    jupyter nbextensions_configurator enable && \
    mkdir -p ${jupyter_data_dir}/nbextensions && \
    mkdir -p ${jupyter_conf_dir}/nbconfig && \
    jupyter notebook --generate-config -y
```

```
    --volume jupyter_config/etc/netrc:/root/.netrc \
    --volume jupyter_config/jupyter:/root/.jupyter \
    --volume jupyter_config/local/share/jupyter:/root/.local/share/jupyter \
```


## Files

- `$(jupyter --config-dir)/nbconfig/notebook.json`
- `$(jupyter --config-dir)/jupyter_notebook_config.json`
- `$(jupyter --data-dir)/nbextensions/vim_binding`

## Pip

- `pip3 install autopep8 ipywidgets`

## Run

- jupyter notebook --no-browser --notebook-dir=/data --allow-root --ip=0.0.0.0 --port=8118


## Config

```json

{
  "NotebookApp": {
    "nbserver_extensions": {
      "jupyter_nbextensions_configurator": true
    },
    "password": "sha1:cd4d85cd9743:1f2cb9b104d49163c10a06c19d1c5281600893dc"
  }
}

```

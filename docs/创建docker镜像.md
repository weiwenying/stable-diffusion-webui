# Stable Diffusion WebUI on BlueWhale Platform

`时间：2024-10-14`  `作者：魏文应`

---

## 简介

本镜像是在博华蓝鲸平台上部署的镜像，仅限博华蓝鲸平台可用。[AUTOMATIC1111](https://github.com/AUTOMATIC1111)/**[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)** 项目在NVIDIA GPU设备上的基本镜像。目前支持SD1.5、SDXL。

| 版本                        | 系统配置                          | 备注 |
| --------------------------- | --------------------------------- | ---- |
| base-cuda12.1-ubuntu20.04   | CUDA12.1.1 Ubuntu20.04 Python3.12 |      |
| v0.0.1-cuda12.1-ubuntu20.04 | CUDA12.1.1 Ubuntu20.04 Python3.10 |      |

使用镜像创建容器：

```bash
CONTAINER_NAME="$USER"-whalesdwebui  # 自定义容器名称
IMAGE_NAME=nuvic/whale-sdwebui:v0.0.1-cuda12.1-ubuntu20.04

# 抛出mount.nfs: Operation not permitted时, 要docker启动参数要加上--privileged
# docker run --gpus 指定物理GPU  -it --name 自定义容器名称 镜像名称:版本(相应版本要求物理机nvidia driver支持) bash
# --gpus '"device=0,1"'
docker run --privileged --shm-size 16G --network host --gpus all -it --name ${CONTAINER_NAME} ${IMAGE_NAME} bash
```

启动后，可以在容器内，执行如下命令测试（`base-cuda12.1-ubuntu20.04` 不适用）：

```bash
# 有多个GPU时，可以指定某个GPU:
# git pull origin nuvic && CUDA_VISIBLE_DEVICES=7 python launch.py --skip-version-check --skip-install --skip-load-model-at-start --no-download-sd-model --xformers --listen --port 12345 --api --ckpt-dir "/userhome/base" --lora-dir "/userhome/lora"  --heartbeat-host 10.1.1.28 --heartbeat-frequency 60

# whale-sdwebui:v0.0.7-cuda12.1-ubuntu20.04的启动命令：
cd /root/workspace/stable-diffusion-webui && ./whale_env_update.sh && ./whale_env.sh && python launch.py --skip-version-check --skip-install --skip-load-model-at-start --no-download-sd-model --xformers --listen --port 12345 --api --share --ckpt-dir "/userhome/base" --lora-dir "/userhome/lora"  --heartbeat-host 10.1.2.6 --heartbeat-frequency 60 --heartbeat-user root --heartbeat-password 12345@iivA --disable-console-progressbars

# whale-sdwebui:v0.0.5-cuda12.1-ubuntu20.04的启动命令：
cd /root/workspace/stable-diffusion-webui && bash ./whale_env.sh nuvic && python launch.py --skip-version-check --skip-install --skip-load-model-at-start --no-download-sd-model --xformers --listen --port 12345 --api --ckpt-dir "/userhome/base" --lora-dir "/userhome/lora"  --heartbeat-host 10.1.1.28 --heartbeat-frequency 60 --share

# whale-sdwebui:v0.0.3-cuda12.1-ubuntu20.04的启动命令：
cd /root/workspace/stable-diffusion-webui && bash ./whale_env.sh whale-sdwebui-v0.0.3 && python launch.py --skip-version-check --skip-install --skip-load-model-at-start --no-download-sd-model --xformers --listen --port 12345 --api --ckpt-dir "/userhome/base" --lora-dir "/userhome/lora"  --heartbeat-host 10.1.1.28 --heartbeat-frequency 60

# whale-sdwebui:v0.0.2-cuda12.1-ubuntu20.04的启动命令：
cd /root/workspace/stable-diffusion-webui && bash ./whale_env.sh && python launch.py --skip-version-check --skip-install --skip-load-model-at-start --no-download-sd-model --xformers --listen --port 12345 --api --ckpt-dir "/userhome/base" --lora-dir "/userhome/lora"  --heartbeat-host 10.1.1.28 --heartbeat-frequency 60

# whale-sdwebui:v0.0.1-cuda12.1-ubuntu20.04的启动命令：
git pull origin nuvic && python launch.py --skip-version-check --skip-install --skip-load-model-at-start --no-download-sd-model --xformers --listen --port 12345 --api --ckpt-dir "/userhome/base" --lora-dir "/userhome/lora"  --heartbeat-host 10.1.1.28 --heartbeat-frequency 60
```

天数智芯GPU:

```bash
# whale-sdwebui:v0.0.8-corex3.2.0-ubuntu20.04的启动命令：
cd /root/workspace/stable-diffusion-webui && ./whale_env_update.sh && ./whale_env.sh && python launch.py --skip-version-check --skip-install --skip-load-model-at-start --no-download-sd-model --listen --port 12345 --api --ckpt-dir "/userhome/base" --lora-dir "/userhome/lora"  --heartbeat-host 10.1.2.6 --heartbeat-frequency 60 --heartbeat-user root --heartbeat-password 12345@iivA --disable-console-progressbars --share
```

启动命令说明：

| 命令/参数                        | 说明                                                    |
| -------------------------------- | ------------------------------------------------------- |
| `git pull origin nuvic`          | 更新代码                                                |
| `python launch.py`               | Stable Diffusion WebUI的启动命令                        |
| `--skip-version-check`           | 跳过torch和xformers版本检测                             |
| `--skip-install`                 | 跳过python packages依赖安装                             |
| `--skip-load-model-at-start`     | 跳过模型启动时自动加载                                  |
| `--no-hashing`                   | 不进行哈希校验，提高加载速度                            |
| `--no-download-sd-model`         | Stable-Diffusion目录一个模型都没有的时候，不自动下载    |
| `--xformers`                     | 启动xformers                                            |
| `--listen`                       | 允许localhost以外的服务连接。                           |
| `--port 12345`                   | 端口                                                    |
| `--api`                          | 启动API，这样才能用POST和GET                            |
| `--ckpt-dir "/base"`             | SD底模型文件所在文件夹，蓝鲸算力中心集群中固定为"/base" |
| `--lora-dir "/lora"`             | LORA模型文件所在文件夹，蓝鲸算力中心集群中固定为"/lora" |
| `--heartbeat-host 10.1.1.28`     | MySQL数据库的IP地址，用于心跳服务                       |
| `--heartbeat-frequency 60`       | 心跳频率                                                |
| `--disable-console-progressbars` | 不显示进度条                                            |

## 创建过程说明

这部分，将给你展示，这些镜像是如何制作的，你不必进行后面的所有操作。之所以提供这些镜像的创建过程，是以便你进一步了解这些镜像的用途，以及当你需要定制化一些操作时，下面这些内容可为你提供参考。

### whale-sdwebui:base-cuda12.1-ubuntu20.04

| 操作                | 备注 |
| ------------------- | ---- |
| 设置工作区间workdir |      |

创建一个Dockerfile文件, 文件内写入如下内容：

```dockerfile
FROM nuvic/miniconda:cuda12.1-ubuntu20.04-jupyterlab4.2.5

WORKDIR /root/workspace/stable-diffusion-webui
```

然后在Dockerfile相同目录下，执行如下命令：

```bash
docker build -t nuvic/whale-sdwebui:base-cuda12.1-ubuntu20.04 ./
```

这样，就创建了一个新镜像 `nuvic/whale-sdwebui:base-cuda12.1-ubuntu20.04` ，并上传到dockerhub中：

```bash
docker push nuvic/whale-sdwebui:base-cuda12.1-ubuntu20.04
# 蓝鲸平台
docker tag nuvic/whale-sdwebui:base-cuda12.1-ubuntu20.04 harbor.bhuhd.com:1443/aigc/whale-sdwebui:base-cuda12.1-ubuntu20.04
```

### whale-sdwebui:v0.0.1-cuda12.1-ubuntu20.04

| 操作                           | 备注                 |
| ------------------------------ | -------------------- |
| Stable Diffusion WebUI运行环境 | 支持`SD 1.5`、`SDXL` |

在物理机上，执行下面命令，启动容器：

```bash
container_name="$USER"-whalesdwebui  # 自定义容器名称
base_image=nuvic/whale-sdwebui:base-cuda12.1-ubuntu20.04
commit_image=nuvic/whale-sdwebui:v0.0.1-cuda12.1-ubuntu20.04

# 注意：不推荐在容器内进行mount操作，因为这样不符合docker环境隔离原则。
# 如果需要在容器内mount, 抛出mount.nfs: Operation not permitted时, 要docker启动参数要加上--privileged
# docker run --gpus 指定物理GPU  -it --name 自定义容器名称 镜像名称:版本(相应版本要求物理机nvidia driver支持) bash
# --gpus '"device=0,1"'
docker run --privileged --network host --gpus all -it --name ${container_name} ${base_image} bash
```

启动容器后，在容器内，执行下面命令，安装配置Python环境：

```bash
conda config --add channels defaults

cd /root/workspace/
git clone -b nuvic http://gitlab.iiva.org.cn/nuvic/2024/stable-diffusion-webui.git
cd /root/workspace/stable-diffusion-webui
git pull origin nuvic

conda install python=3.10

# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

conda install platformdirs tomli streamlit

# https://github.com/facebookresearch/xformers
# xformers的版本要求比较严格，安装时要注意cuda相关库是否在下载，如果在下载，说明版本和当前Pytorch版本不匹配
conda install xformers -c xformers

# ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found 
# 不推荐修改LD_LIBRARY_PATH，不让curl命令等还会抛出类似.so.6异常
# export LD_LIBRARY_PATH=/root/miniconda3/lib:$LD_LIBRARY_PATH
# 而是切换pandas版本（我这里是pandas版本导致的，如果你是其它包，则切换为其它包）
pip install pandas==2.2.3

# 然后 `requirements_versions.txt` 打开文件，删除里面的 `torch`
pip install -r requirements_versions.txt
pip install "httpx>=0.25.0"
pip install mysql-connector-python
pip install git+https://github.com/openai/CLIP.git

# bash: /root/miniconda3/lib/libtinfo.so.6: no version information available 
conda install -c conda-forge ncurses

# 成功执行会显示 successfully
jupyter-lab --no-browser
```

接着，挂载共享文件夹：

```bash
# 加上nfs-kernel-server，解决mount.nfs: Protocol not supported
apt update && apt install -y nfs-common cifs-utils freeipa-client-samba nfs-kernel-server sudo

# 前缀10.1.252.1:/ds_fs/n/你的共享文件夹 /本机目录
mkdir -p /base && mount -t nfs 10.1.252.1:/ds_fs/n/public/models/sd/base /base
mkdir -p /lora && mount -t nfs 10.1.252.1:/ds_fs/n/public/models/sd/lora /lora
```

执行下面命令：

```bash
vim /etc/hosts
```

添加如下代码：

```bash
10.1.252.5    svn.iiva.org.cn
10.1.252.5    qnap.iiva.org.cn
10.1.252.5    gitlab.iiva.org.cn
```

之后就可以启动webui了：

```bash
python launch.py --skip-version-check  --skip-install --skip-load-model-at-start --no-hashing --no-download-sd-model --xformers --listen --port 12345 --api --ckpt-dir "/base" --lora-dir "/lora"  --heartbeat-host 10.1.1.28 --heartbeat-frequency 1
```

启动后，在浏览器中确认webui无误后，清除缓存，并退出容器：

```bash
# 清除缓存
apt autoclean && apt clean && apt autoremove
pip cache purge && conda clean -y --all
exit
```

将容器提交为镜像，上传镜像到远程仓库：

```bash
# docker commit 容器名称 镜像名称
docker commit ${container_name} ${commit_image}

# 上传到官方hub.docker.com
# docker login
docker push ${commit_image}

# 上传到博华，先登录博华的harbor 10.1.2.1:1443
# docker login harbor.bhuhd.com:1443
bohua_push=harbor.bhuhd.com:1443/aigc/whale-sdwebui:v0.0.1-cuda12.1-ubuntu20.04
docker tag ${commit_image} ${bohua_push}

# 上传镜像
# docker push 刚才你build的镜像
docker push ${bohua_push}
```

平台启动命令可以如下：

```bash
git pull origin nuvic && python launch.py --skip-version-check  --skip-install --skip-load-model-at-start --no-download-sd-model --xformers --listen --port 12345 --api --ckpt-dir "/base" --lora-dir "/lora"  --heartbeat-host 10.1.1.28 --heartbeat-frequency 1
```

> Tips: 分布式存储挂载命令：
>
> ```bash
> # 前缀10.1.252.1:/ds_fs/n/你的共享文件夹 /本机目录
> mkdir -p /root/workspace/stable-diffusion-webui/models//Stable-diffusion
> mount -t nfs 10.1.252.1:/ds_fs/n/public/models/sd/lora /root/workspace/stable-diffusion-webui/models/Stable-diffusion
> 
> mkdir -p /root/workspace/stable-diffusion-webui/models/Lora
> mount -t nfs 10.1.252.1:/ds_fs/n/public/models/sd/lora /root/workspace/stable-diffusion-webui/models/Lora
> ```
>
> k8s平台上，容器内是无法mount的，这时候，只能用分布式存储 `/userhome` 目录了，每个用户一个这样的目录，目录如下：
>
> ```bash
> # sudo mount -t nfs 10.1.252.1:/ds_fs/n/octopus/data/minio/95189cbd28714b1f987b3bbe1cece183/userhome /userhome
> # william这个用户，这个目录容器启动的时候，平台自动挂载到容器内的/userhome目录
> 10.1.252.1:/ds_fs/n/octopus/data/minio/95189cbd28714b1f987b3bbe1cece183/userhome
> ```
>
> k8s平台上：
>
> ```bash
> git pull origin nuvic && python launch.py --skip-version-check  --skip-install --skip-load-model-at-start --no-download-sd-model --xformers --listen --port 12345 --api --ckpt-dir "/userhome/base" --lora-dir "/userhome/lora"  --heartbeat-host 10.1.1.28 --heartbeat-frequency 1
> ```

### whale-sdwebui:v0.0.2-cuda12.1-ubuntu20.04

在物理机上，执行下面命令，启动容器：

```bash
# 前缀10.1.252.1:/ds_fs/n/你的共享文件夹 /本机目录
sudo mkdir -p /base && sudo mount -t nfs 10.1.252.1:/ds_fs/n/public/models/sd/lora /base
sudo mkdir -p /lora && sudo mount -t nfs 10.1.252.1:/ds_fs/n/public/models/sd/lora /lora

container_name="$USER"-whalesdwebui  # 自定义容器名称
base_image=nuvic/whale-sdwebui:v0.0.1-cuda12.1-ubuntu20.04
commit_image=nuvic/whale-sdwebui:v0.0.2-cuda12.1-ubuntu20.04

# 注意：不推荐在容器内进行mount操作，因为这样不符合docker环境隔离原则。
# 如果需要在容器内mount, 抛出mount.nfs: Operation not permitted时, 要docker启动参数要加上--privileged
# docker run --gpus 指定物理GPU  -it --name 自定义容器名称 镜像名称:版本(相应版本要求物理机nvidia driver支持) bash
# --gpus '"device=0,1"'
docker run --privileged --network host --gpus all -v /base:/base -v /lora:/lora -it --name ${container_name} ${base_image} bash
```

然后在容器内置执行下面命令，设置为 [清华pip源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) ：

```bash
# pip config set global.index-url 清华pip源
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

然后将 `/etc/apt/sources.list` 内容修改如下，使用 [清华Ubuntu 软件仓库](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)：

```bash
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
```

然后同步代码：

```bash
git pull origin whale-sdwebui-v0.0.2
```

启动后，在浏览器中确认webui无误后，清除缓存，并退出容器：

```bash
# 清除缓存
apt autoclean && apt clean && apt autoremove
pip cache purge && conda clean -y --all
exit
```

在物理机，执行下面命令，测试一下：

```bash
docker start ${container_name}

docker exec ${container_name} bash -c "bash ./whale_env.sh && python launch.py --skip-version-check --skip-install --skip-load-model-at-start --no-download-sd-model --xformers --listen --port 12345 --api --ckpt-dir '/base' --lora-dir '/lora'  --heartbeat-host 10.1.1.28 --heartbeat-frequency 60"

docker stop  ${container_name}
```

将容器提交为镜像，上传镜像到远程仓库：

```bash
# docker commit 容器名称 镜像名称
docker commit ${container_name} ${commit_image}

# 上传到官方hub.docker.com
# docker login
docker push ${commit_image}

# 上传到博华，先登录博华的harbor 10.1.2.1:1443
# docker login harbor.bhuhd.com:1443
bohua_push=harbor.bhuhd.com:1443/aigc/whale-sdwebui:v0.0.2-cuda12.1-ubuntu20.04
docker tag ${commit_image} ${bohua_push}

# 上传镜像
# docker push 刚才你build的镜像
docker push ${bohua_push}
```

### whale-sdwebui:v0.0.3-cuda12.1-ubuntu20.04

更新了stable diffusion webui的 `whale_env.sh` 脚本。

### whale-sdwebui:v0.0.4-cuda12.1-ubuntu20.04

```bash
container_name="$USER"-whalesdwebui  # 自定义容器名称
base_image=nuvic/whale-sdwebui:v0.0.3-cuda12.1-ubuntu20.04
commit_image=nuvic/whale-sdwebui:v0.0.4-cuda12.1-ubuntu20.04

docker run --privileged --network host --gpus all -v /base:/base -v /lora:/lora -it --name ${container_name} ${base_image} bash

# 要启动科学上网代理工具，下载一个文件，在容器内执行
wget https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64 && \
mv frpc_linux_amd64 frpc_linux_amd64_v0.2 && \
chmod +x frpc_linux_amd64_v0.2 && \
mv frpc_linux_amd64_v0.2 /root/miniconda3/lib/python3.10/site-packages/gradio/

python launch.py --skip-version-check --skip-install --skip-load-model-at-start --no-download-sd-model --xformers --listen --share --port 12345 --api --ckpt-dir "/base" --lora-dir "/lora"  --heartbeat-host 10.1.1.28 --heartbeat-frequency 60
# 退出
exit

# 物理机执行提交镜像
# docker commit 容器名称 镜像名称
docker commit ${container_name} ${commit_image}
# 上传到官方hub.docker.com: docker login
docker push ${commit_image}
# 上传到博华，先登录博华的harbor 10.1.2.1:1443: docker login harbor.bhuhd.com:1443
bohua_push=harbor.bhuhd.com:1443/aigc/whale-sdwebui:v0.0.4-cuda12.1-ubuntu20.04
docker tag ${commit_image} ${bohua_push}
docker push ${bohua_push}
```

### whale-sdwebui:v0.0.5-cuda12.1-ubuntu20.04

更新了stable diffusion webui的 `whale_env.sh` 脚本。

### whale-sdwebui:v0.0.6-cuda12.1-ubuntu20.04

更新了stable diffusion webui的 `whale_env.sh` 脚本。

```bash
container_name="$USER"-whalesdwebui  # 自定义容器名称
base_image=nuvic/whale-sdwebui:v0.0.5-cuda12.1-ubuntu20.04
commit_image=nuvic/whale-sdwebui:v0.0.6-cuda12.1-ubuntu20.04
# 上传到博华，先登录博华的harbor 10.1.2.1:1443: docker login harbor.bhuhd.com:1443
bohua_push=harbor.bhuhd.com:1443/aigc/whale-sdwebui:v0.0.6-cuda12.1-ubuntu20.04

docker run --privileged --network host --gpus all -v /base:/base -v /lora:/lora -it --name ${container_name} ${base_image} bash

git remote add iiva http://gitlab.iiva.org.cn/nuvic/2024/stable-diffusion-webui.git
git remote remove origin
git remote add origin https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
git checkout nuvic && git pull iiva nuvic

# 物理机执行提交镜像
# docker commit 容器名称 镜像名称
docker commit ${container_name} ${commit_image}
# 上传到官方hub.docker.com: docker login
docker push ${commit_image}
docker tag ${commit_image} ${bohua_push}
docker push ${bohua_push}
docker rm ${container_name}
```

### whale-sdwebui:v0.0.7-cuda12.1-ubuntu20.04

更新了stable diffusion webui的 `whale_env_update.sh` 脚本。

### whale-sdwebui:v0.0.8-cuda12.1-ubuntu20.04

更新了stable diffusion webui的 `whale_env_update.sh` 脚本。

```bash
git remote remove iiva
git remote remove origin
git remote add origin http://gitlab.iiva.org.cn/nuvic/2024/stable-diffusion-webui.git
git checkout nuvic && git pull origin nuvic
```

### whale-sdwebui:v0.0.8-corex3.2.0-ubuntu20.04

在天数智芯GPU服务器上，执行下面命令：

```bash
docker tag nuvic/whale-sdwebui:3.2.0-whale-base nuvic/whale-sdwebui:v0.0.8-corex3.2.0-ubuntu20.04
```

### ~~whale-sdwebui:base-corex3.2.1-ubuntu20.04~~

在天数智芯GPU服务器上，执行下面命令：

```bash
# docker login 10.1.2.1:1443
container_name="$USER"-whalesdwebui  # 自定义容器名称
base_image=nuvic/corex:3.2.1-whale-base
commit_image=nuvic/whale-sdwebui:base-corex3.2.1-ubuntu20.04
bohua_push=harbor.bhuhd.com:1443/aigc/whale-sdwebui:base-corex3.2.1-ubuntu20.04

docker run -it --privileged --cap-add=ALL --pid=host --network host \
-v /usr/src:/usr/src \
-v /lib/modules:/lib/modules \
-v /dev:/dev \
--name ${container_name} ${base_image} bash
```

执行下面命令，配置webui环境：

```bash
echo "10.1.252.5  gitlab.iiva.org.cn"  >> /etc/hosts

pip install git+http://gitlab.iiva.org.cn/nuvic/2024/aigcapi.git@mysql

cd /root/workspace/
git clone -b nuvic http://gitlab.iiva.org.cn/nuvic/2024/stable-diffusion-webui.git
cd /root/workspace/stable-diffusion-webui
git pull origin nuvic

pip install platformdirs tomli streamlit
# 天数无法使用xformers
# conda install xformers -c xformers
pip install -r requirements_versions_corex.txt
# TypeError: AsyncConnectionPool.__init__() got an unexpected keyword argument 'socket_options'
pip install httpx==0.25.0
pip install mysql-connector-python
pip install git+https://github.com/openai/CLIP.git
pip install -U huggingface_hub
# pip install transformers>=4.38.0

# 成功执行会显示 successfully
jupyter-lab --no-browser --allow-root
```

python依赖配置好之后，挂载测试模型：

```bash
# 前缀10.1.252.1:/ds_fs/n/你的共享文件夹 /本机目录
mkdir -p /base && mount -t nfs 10.1.252.1:/ds_fs/n/public/models/sd/base /base
mkdir -p /lora && mount -t nfs 10.1.252.1:/ds_fs/n/public/models/sd/lora /lora
```

启动webui测试：

```bash
python launch.py --skip-version-check  --skip-install --skip-load-model-at-start --no-download-sd-model --listen --port 12345 --api --ckpt-dir "/base" --lora-dir "/lora"  --heartbeat-host 10.1.1.28 --heartbeat-frequency 1
```

```bash
pip install mysql-connector-python
pip install pytorch_lightning==1.9.4
pip install gradio==3.41.2
pip install omegaconf==2.2.3
```

```bash
transformers>=4.38.0
python launch.py --skip-version-check  --skip-install --skip-load-model-at-start --no-download-sd-model --listen --port 12345 --api --heartbeat-host 10.1.1.28 --heartbeat-frequency 1
```

```bash
A tensor with all NaNs was produced in VAE.
Web UI will now convert VAE into bfloat16 and retry.
To disable this behavior, disable the 'Automatically convert VAE to bfloat16' setting.
```

```bash
modules.devices.NansException: A tensor with NaNs was produced in Unet. This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the "Upcast cross attention layer to float32" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this. Use --disable-nan-check commandline argument to disable this check.
```

```bash
 python launch.py --skip-version-check  --skip-install --skip-load-model-at-start --no-download-sd-model --listen --port 12345 --api --heartbeat-host 10.1.1.28 --heartbeat-frequency 1 --no-half-vae --no-half --precision full --medvram-sdxl
```

```bash
 python launch.py --skip-version-check  --skip-install --skip-load-model-at-start --no-download-sd-model --listen --port 12345 --api --heartbeat-host 10.1.1.28 --heartbeat-frequency 1 --precision half
```




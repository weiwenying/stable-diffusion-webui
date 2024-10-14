# NUVIC BRANCH

## 测试

下载源代码和安装Python依赖：

```bash
git clone git@github.com:weiwenying/stable-diffusion-webui.git
cd stable-diffusion-webui-forge/
git checkout v1.10.1 -b nuvic

conda create -n noflux python=3.10
conda activate noflux

# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

conda install platformdirs tomli streamlit

# https://github.com/facebookresearch/xformers
# xformers的版本要求比较严格，安装时要注意cuda相关库是否在下载，如果在下载，说明版本和当前Pytorch版本不匹配
conda install xformers -c xformers

# 然后 `requirements_versions.txt` 打开文件，删除里面的 `torch`
pip install -r requirements_versions.txt

python launch.py --skip-version-check --skip-prepare-environment --skip-install --xformers --listen --api --port 12345
```

## 修改代码

### Lora绝对路径支持

默认的lora文件夹为：

```
stable-diffusion-webui/models/Lora
```

默认Lora遍历方式为，将上述Lora文件夹遍历一遍，找到下面的后缀文件：

```python
[".pt", ".ckpt", ".safetensors"]
```

当你的lora目录内，有非常多lora文件，同时lora文件会动态删减。这时候，当你添加一个Lora文件时候，就要遍历整个目录，这个效率太低了。因此，你可以修改 `extensions-builtin/Lora/networks.py` 文件，添加如下内容：

```python

def check_path_or_name(names):
    filepaths = []
    filenames = []
    filenames_ = []
    for n in names:
        x = os.path.splitext(os.path.basename(n))[0]
        if x == n:
            filenames.append(x)
        else:
            filenames_.append(x)
            filepaths.append(n)
    return (filenames_, filepaths), filenames


def load_lora_from_path(filenames_, filepaths):
    """weiwenying520@gmail.com
    """
    networks_on_disk = []
    failed_to_load_networks = []
    for name, path in zip(filenames_, filepaths):
        try:
            # name = 'datasetw-000001'
            # filename = '/home/william/projects/public/stable-diffusion-webui/models/Lora/datasetw-000001.safetensors'
            if not os.path.exists(path):
                failed_to_load_networks.append(path)
                raise OSError(f"Failed to load network {name} from {path}")
            entry = network.NetworkOnDisk(name, path)
            networks_on_disk.append(entry)
        except OSError:  # should catch FileNotFoundError and PermissionError etc.
            errors.report(f"Failed to load network {name} from {path}", exc_info=True)
            continue
    return networks_on_disk, failed_to_load_networks


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    emb_db = sd_hijack.model_hijack.embedding_db
    already_loaded = {}

    for net in loaded_networks:
        if net.name in names:
            already_loaded[net.name] = net
        for emb_name, embedding in net.bundle_embeddings.items():
            if embedding.loaded:
                emb_db.register_embedding_by_name(None, shared.sd_model, emb_name)

    loaded_networks.clear()

    (filenames_, filepaths), names = check_path_or_name(names)
    networks_on_disk_, failed_to_load_networks = load_lora_from_path(filenames_, filepaths)

    unavailable_networks = []
    for name in names:
        if name.lower() in forbidden_network_aliases and available_networks.get(name) is None:
            unavailable_networks.append(name)
        elif available_network_aliases.get(name) is None:
            unavailable_networks.append(name)

    if unavailable_networks:
        update_available_networks_by_names(unavailable_networks)

    networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()

        networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]

    networks_on_disk.extend(networks_on_disk_)  # weiwenying520@gmail.com
    names.extend(filenames_)
    
    # failed_to_load_networks = []

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        net = already_loaded.get(name, None)

        if network_on_disk is not None:
            if net is None:
                net = networks_in_memory.get(name)

            if net is None or os.path.getmtime(network_on_disk.filename) > net.mtime:
                try:
                    net = load_network(name, network_on_disk)

                    networks_in_memory.pop(name, None)
                    networks_in_memory[name] = net
                except Exception as e:
                    errors.display(e, f"loading network {network_on_disk.filename}")
                    continue

            net.mentioned_name = name

            network_on_disk.read_hash()

        if net is None:
            failed_to_load_networks.append(name)
            logging.info(f"Couldn't find network with name {name}")
            continue

        net.te_multiplier = te_multipliers[i] if te_multipliers else 1.0
        net.unet_multiplier = unet_multipliers[i] if unet_multipliers else 1.0
        net.dyn_dim = dyn_dims[i] if dyn_dims else 1.0
        loaded_networks.append(net)

        for emb_name, embedding in net.bundle_embeddings.items():
            if embedding.loaded is None and emb_name in emb_db.word_embeddings:
                logger.warning(
                    f'Skip bundle embedding: "{emb_name}"'
                    ' as it was already loaded from embeddings folder'
                )
                continue

            embedding.loaded = False
            if emb_db.expected_shape == -1 or emb_db.expected_shape == embedding.shape:
                embedding.loaded = True
                emb_db.register_embedding(embedding, shared.sd_model)
            else:
                emb_db.skipped_embeddings[name] = embedding

    if failed_to_load_networks:
        lora_not_found_message = f'Lora not found: {", ".join(failed_to_load_networks)}'
        sd_hijack.model_hijack.comments.append(lora_not_found_message)
        if shared.opts.lora_not_found_warning_console:
            print(f'\n{lora_not_found_message}\n')
        if shared.opts.lora_not_found_gradio_warning:
            gr.Warning(lora_not_found_message)

    purge_networks_from_memory()
```

这样，就可以在Prompt中，以：

```bash
# 文字描述 <lora:Lora模型名称:1>   # 默认写法
一只黄毛猫咪 <lora:datasetw-000001:1>

# 文字描述 <lora:/绝对路径/Lora模型名称:1>    # 优化后的带路径写法
一只黄毛猫咪 <lora:/root/workspace/stable-diffusion-webui/models/Lora/datasetw-000001:1>
```


SDWEBUI_VERSION=${1:-"nuvic"}

echo $SDWEBUI_VERSION

pip uninstall aigcapi -y

case "$SDWEBUI_VERSION" in
    "whale-sdwebui-v0.0.2" | "whale-sdwebui-v0.0.3" | "whale-sdwebui-v0.0.4")
        echo "10.1.252.5  gitlab.iiva.org.cn"  >> /etc/hosts
        pip install git+http://gitlab.iiva.org.cn/nuvic/2024/aigcapi.git@mysql
        git pull origin nuvic
        git fetch --tags
        git checkout -b runtime ${1:-"nuvic"}
        ;;
    *)
        pip install git+http://gitlab.iiva.org.cn/nuvic/2024/aigcapi.git@mysql
        git fetch --tags
        git checkout -b runtime ${1:-"nuvic"}
        ;;
esac

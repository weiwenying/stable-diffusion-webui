echo "10.1.252.5  gitlab.iiva.org.cn"  >> /etc/hosts

pip install git+http://gitlab.iiva.org.cn/nuvic/2024/aigcapi.git@mysql

git checkout nuvic && git pull iiva nuvic

git fetch --tags

git checkout -b runtime ${1:-"nuvic"}

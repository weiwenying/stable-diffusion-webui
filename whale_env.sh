echo "10.1.252.5  gitlab.iiva.org.cn"  >> /etc/hosts

pip install git+http://gitlab.iiva.org.cn/nuvic/2024/aigcapi.git@mysql

git pull origin nuvic

git fetch --tags

git checkout -b runtime ${1:-"nuvic"}

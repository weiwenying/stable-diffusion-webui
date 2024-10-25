echo "10.1.252.5  gitlab.iiva.org.cn"  >> /etc/hosts

# wget http://gitlab.iiva.org.cn/nuvic/2024/stable-diffusion-webui/-/raw/nuvic/whale_env.sh?inline=false -O whale_env.sh
git checkout nuvic && git pull iiva nuvic

# chmod +x ./whale_env.sh && ./whale_env.sh $@
chmod +x ./whale_env.sh 

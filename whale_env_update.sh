
echo "启动whale_env_pudate.sh"
echo "10.1.252.5  gitlab.iiva.org.cn"  >> /etc/hosts
echo "git checkout nuvic && git pull origin nuvic"
# wget http://gitlab.iiva.org.cn/nuvic/2024/stable-diffusion-webui/-/raw/nuvic/whale_env.sh?inline=false -O whale_env.sh
git checkout nuvic && git pull origin nuvic

# chmod +x ./whale_env.sh && ./whale_env.sh $@
chmod +x ./whale_env.sh 
echo "whale_env_pudate.sh完毕"

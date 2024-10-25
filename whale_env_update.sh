
printf "\n\n更新脚本(whale_env_pudate.sh):\n"
echo "10.1.252.5  gitlab.iiva.org.cn"  >> /etc/hosts
echo "git checkout nuvic && git pull origin nuvic"
# wget http://gitlab.iiva.org.cn/nuvic/2024/stable-diffusion-webui/-/raw/nuvic/whale_env.sh?inline=false -O whale_env.sh
git checkout nuvic && git pull origin nuvic

# chmod +x ./whale_env.sh && ./whale_env.sh $@
chmod +x ./whale_env.sh 
printf "脚本更新完成\n\n"

## 部署信息
- 隧道 URL：https://zsbot.xxx.trycloudflare.com
- 本地端口：8000
- 隧道配置：Cloudflare Zero Trust → Tunnels → zhuanshengben-bot

## 部署步骤
1. 注册 Cloudflare Zero Trust（免费）
2. 创建 Tunnel，选择 Cloudflared
3. Windows 安装 cloudflared.msi
4. 执行安装命令，配置 Public Hostname 指向 127.0.0.1:8000
5. 启动 uvicorn，隧道自动连接

## 已知限制
- 电脑关机 = 服务下线
- W7 上香港服务器后解决

##使用方法
先启动docker容器，在在终端启动uvicorn app.main:app --host 127.0.0.1 --port 8000，最后在终端输入cloudflared tunnel --url http://127.0.0.1:8000访问临时网站

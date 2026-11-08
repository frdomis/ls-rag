# 服务器部署信息（W7D1）

## 基本信息
- 云厂商：腾讯云轻量应用服务器
- 地域：中国香港
- 配置：2 核 2G / 40GB SSD / 20Mbps
- 公网 IP：43.128.31.138
- 系统：Ubuntu 22.04 LTS
- 购买日期：2026-11-02
- 到期：2026-12-02（1个月，需手动续费）

## SSH 登录
ssh -i $env:USERPROFILE\.ssh\lsrag root@43.128.31.138

## 服务管理
systemctl status lsrag # 查状态
systemctl restart lsrag # 重启
systemctl stop lsrag # 停止
tail -f /var/log/lsrag.log # 看日志

## Docker / Qdrant
docker ps # 看容器
docker restart qdrant # 重启 Qdrant
docker logs --tail 50 qdrant # Qdrant 日志
curl -s http://localhost:6333/collections # 验证

## 目录
/opt/ls-rag/ # 代码
/opt/ls-rag/.env # 环境变量（含 API Key）
/opt/ls-rag/venv/ # Python 虚拟环境
/opt/qdrant/ # Qdrant 数据（挂载）
/var/log/lsrag.log # 应用日志

## 访问地址
http://43.128.31.138/health # 健康检查
http://43.128.31.138/docs # Swagger UI
http://43.128.31.138/chat # 问答
http://43.128.31.138/search # 检索

## 端口
22 # SSH（对外开放）
80 # Nginx（对外开放）
8000 # API（仅 127.0.0.1，通过 Nginx 转发）
6333 # Qdrant（仅 localhost）

## 关键数字
- 语料：142 块
- 向量：1024 维（BGE-M3）
- 同源校验：142 点 == 142 块 ✅

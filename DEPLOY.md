# CV-Agent 云端部署说明

## 推荐方式：Render Web Service

1. 把 `cv-agent` 目录上传到 GitHub 仓库。
2. 在 Render 新建 `Web Service`，选择这个仓库。
3. 配置：
   - Runtime: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.server:app --host 0.0.0.0 --port $PORT`
4. 在 Render 的 Environment Variables 里添加：
   - `DEEPSEEK_API_KEY`: 你的 DeepSeek API Key
   - `DEEPSEEK_BASE_URL`: `https://api.deepseek.com/v1`
   - `DEEPSEEK_MODEL_CHAT`: `deepseek-chat`
   - `CONF_LOW_FALLBACK`: 兜底话术，可不填，代码里已有默认值
5. 部署完成后，Render 会给一个公网 HTTPS 地址，例如 `https://xxx.onrender.com`。

## 注意

- 不要把 `.env` 上传到 GitHub，里面有 API Key；本项目已加入 `.gitignore`。
- `data/raw/` 里的原始简历 PDF、截图不会被 Docker 镜像包含；线上只需要 `data/kb/`。
- 如果平台提示端口问题，确认启动命令里用了 `$PORT`，不要写死 8000。
- 如果 15 分钟没人访问后第一次打开较慢，通常是免费实例休眠导致。

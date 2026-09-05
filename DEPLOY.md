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
   - `ADMIN_PASSWORD`: 管理员后台密码，用于访问 `/admin`
   - `DATABASE_URL`: 免费云 Postgres 连接串，用来持久保存后台提问和模型回复
5. 部署完成后，Render 会给一个公网 HTTPS 地址，例如 `https://xxx.onrender.com`。
6. 管理员后台地址为 `https://xxx.onrender.com/admin`，普通访客不知道密码时看不到用户提问记录。

## 免费持久数据库

Render 免费 Web Service 的本地文件不适合保存后台记录，服务重启或重新部署后 SQLite 记录可能丢失。建议使用 Neon 免费 Postgres：

1. 打开 Neon，新建一个免费的 Postgres 项目。
2. 复制连接串，格式类似 `postgresql://...?...sslmode=require`。
3. 打开 Render 当前服务的 Environment Variables。
4. 新增 `DATABASE_URL`，值粘贴 Neon 的连接串。
5. 保存后重新部署。进入 `/admin`，当前存储显示为“云数据库 Postgres（持久）”就说明成功。

如果没有配置 `DATABASE_URL`，程序会自动回退到本地 SQLite，方便本地开发，但线上记录不保证长期保存。

## 注意

- 不要把 `.env` 上传到 GitHub，里面有 API Key；本项目已加入 `.gitignore`。
- `data/raw/` 里的原始简历 PDF、截图不会被 Docker 镜像包含；线上只需要 `data/kb/`。
- 如果平台提示端口问题，确认启动命令里用了 `$PORT`，不要写死 8000。
- 如果 15 分钟没人访问后第一次打开较慢，通常是免费实例休眠导致。

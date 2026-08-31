# DreamFace HTTP Service

基于 `diffusers.Flux2KleinPipeline` 和 `DreamFace2.0` 的人像一致性 HTTP 服务。

## 启动

建议使用最新版 diffusers：

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

启动服务：

```bash
export DIFFUSERS_MODEL_BASE_PATH=/mnt/model
python start_server.py --port=9001 --debug=False
```

默认模型：

```text
hithink-image-labs/DreamFace2.0
```

## 双卡分布式部署

两块 24G 显卡可以通过 diffusers/Accelerate 的 `device_map` 自动分配 pipeline 组件。该模式会在加载时挂载跨设备 hook，不要再手动把组件 `.to()` 到不同 GPU：

```bash
python start_server.py \
  --port=9001 \
  --debug=False \
  --device_map=balanced \
  --max_memory=0:22GiB,1:22GiB,cpu:60GiB \
  --enable_cpu_offload=False
```

`device_map` 与 `enable_cpu_offload` 互斥；如果同时设置，服务会优先使用 `device_map` 并打印忽略 CPU offload 的提示。启动日志会打印 `text_encoder`、`transformer`、`vae` 的实际设备信息。

## 24G 显存配置

单张 24G 显卡建议使用 diffusers 的 model CPU offload：

```bash
python start_server.py --port=9001 --debug=False --enable_cpu_offload=True
```

也可以降低默认分辨率进一步节省显存：

```bash
python start_server.py \
  --port=9001 \
  --debug=False \
  --enable_cpu_offload=True \
  --default_height=1024 \
  --default_width=768
```

## 接口

请求地址：

```text
POST /image/dreamface
```

JSON 参数：

```json
{
  "prompt": "编辑提示词",
  "pics": ["base64_img_1", "base64_img_2", "base64_img_3"],
  "seed": 42,
  "steps": 4,
  "cfg": 1.0,
  "height": 1152,
  "width": 896
}
```

说明：

- `prompt` 必填。
- `pics` 可选，最多 3 张参考图。
- `pic` 可作为单张参考图字段使用；当 `pics` 存在时优先使用 `pics`。
- 不传 `pic/pics` 时按 text-to-image 调用。
- 返回图片为 `data.img` 中的 base64 PNG。

成功返回：

```json
{
  "code": 0,
  "msg": "Success",
  "data": {
    "img": "base64_png"
  }
}
```

## 前端接入

健康检查：

```text
GET http://<host>:9001/readiness
```

生成接口：

```text
POST http://<host>:9001/image/dreamface
```

浏览器调用示例：

```javascript
async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      const value = String(reader.result);
      resolve(value.includes(",") ? value.split(",")[1] : value);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function callDreamFace({ baseUrl, prompt, files }) {
  const pics = await Promise.all(files.map(fileToBase64));
  const response = await fetch(`${baseUrl}/image/dreamface`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt,
      pics,
      seed: 42,
      steps: 4,
      cfg: 1.0,
      height: 1152,
      width: 896,
    }),
  });

  const result = await response.json();
  if (result.code !== 0) {
    throw new Error(result.msg || "DreamFace request failed");
  }

  return `data:image/png;base64,${result.data.img}`;
}
```

页面使用示例：

```javascript
const resultUrl = await callDreamFace({
  baseUrl: "http://127.0.0.1:9001",
  prompt: "make the person look natural, high quality",
  files: Array.from(document.querySelector("#images").files),
});

document.querySelector("#result").src = resultUrl;
```

说明：

- `pics` 最多传 3 张图片。
- `pics` 建议传不带 `data:image/png;base64,` 前缀的 base64 字符串。
- 返回图片在 `result.data.img` 中，是 base64 PNG。
- 服务已支持 CORS，浏览器前端可以直接调用。

## 请求日志

默认会记录每次 `/image/dreamface` 请求的参数、输入图片、输出图片和错误信息。

```text
logs/requests/
  2026-05-15.jsonl
  images/
    2026-05-15/
      <request_id>/
        input_1.png
        output.png
```

日志配置：

```text
enable_request_log = True
request_log_dir = "logs/requests"
save_request_images = True
save_result_images = True
```

JSONL 中不会记录完整 base64，只记录 prompt、参数、客户端 IP、图片路径、耗时、状态和错误信息，避免日志文件过大。

## 测试

```bash
python conn_test.py test_imgs output
```

可通过环境变量覆盖测试参数：

```bash
export DREAMFACE_PROMPT="A cinematic black-and-white portrait of the same woman, facing directly toward the camera, front-facing view, looking straight at the viewer, sitting elegantly on the floor against a dark background. She wears a long black trench coat, short shorts, and thigh-high leather boots. One leg is bent in a relaxed way while the other extends slightly, her upper body leans slightly forward, one hand gently resting near her face, fingers lightly touching her cheek, while the other hand rests naturally on her thigh. Her wavy hair flows smoothly over her shoulders, statement earrings and a chain necklace. Dramatic high-contrast studio lighting with soft rim light against a seamless dark background, medium full shot with film grain texture"
export DREAMFACE_STEPS=4
export DREAMFACE_CFG=1.0
python conn_test.py test_imgs output
```

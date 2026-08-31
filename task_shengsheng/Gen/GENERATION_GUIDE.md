# 10,000 条手部异常 Prompt 生成说明

## 本批次结果

- 主文件：`prompts_10000.json`
- 数据格式：JSON 数组，每条记录只有 `prompt` 字段
- 总数与唯一数：10,000 / 10,000
- 高质量参考样例：`prompt_samples_10.json` 的 10 条已原样放在主文件最前面
- 场景：50 个基础场景 × 10 个环境版本 = 500 个场景描述
- 性别：男性 5,000，女性 5,000
- 异常范围：单侧 7,000，双侧 3,000
- 构图：普通中景 3,334，三分之二中景 3,333，全身 3,333（10,000 无法被 3 整除，这是最接近严格 1:1:1 的整数分配）
- 异常家族：10 类，每类 1,000 条

## 自然构图版本

`prompts_10000_natural_composition.json` 是后续生成的自然构图版本，原始
`prompts_10000.json` 保持不变。该版本不再统一追加“手部位于前景”“靠近镜头”
“手部锐利对焦”或“全身构图时手部置于身体前侧”等句子；人物取景、场景动作和
异常结构直接组合，使手型描述自然嵌入人物行为。该批次与原批次交集为 0。

## 异常体系

原有 6 类：多指、少指、多手、并指、额外前臂/手臂、双掌结构。

新增 4 类：分叉手指、拇指结构异常、中央裂手、巨指。新增描述位于
`configs/extended_anomalies.yaml`，每种异常都分别定义了单侧和双侧的明确结构描述。

## 文件用途

- `generate_prompts.py`：确定性生成、配额控制和跨批次去重脚本
- `scene_catalog_500.json`：500 个场景描述及其基础场景映射
- `generation_summary.json`：当前批次的实际分布统计
- `prompt_history_sha256.txt`：已生成 prompt 的 SHA-256 指纹，一行一个；续生成时用于阻止跨批次重复
- `configs/extended_anomalies.yaml`：新增异常家族配置

## 以后继续生成且避免重复

在 `Gen` 目录的上级项目路径运行：

```bash
python task_shengsheng/Gen/generate_prompts.py \
  --skip-samples \
  --seed 20260708 \
  --output task_shengsheng/Gen/prompts_10000_batch_02.json \
  --summary task_shengsheng/Gen/generation_summary_batch_02.json
```

必须遵守：

1. 新批次使用新的 `--seed` 和新的 `--output` 文件名。
2. 保留并继续使用同一个 `prompt_history_sha256.txt`，脚本会读取旧指纹并追加新指纹。
3. 后续批次使用 `--skip-samples`，否则最初的 10 条参考样例会被有意再次放入新文件。
4. 不要删除历史指纹文件；删除后只能保证单个文件内部不重复，不能保证跨批次不重复。
5. 若修改场景或异常配置，生成后仍需检查 `generation_summary*.json` 中的数量和分布。

## 质量约束

Prompt 保留手部清晰可见、异常结构精确、单侧异常时另一只手正常、真实摄影质感等要求；不包含“手不得被遮挡”之类的强制约束。动作始终从各场景的 `allowed_actions` 中选择。

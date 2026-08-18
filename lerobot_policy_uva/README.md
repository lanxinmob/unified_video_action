# LeRobot Policy UVA

`lerobot_policy_uva` 是原始 UVA 的 LeRobot 接入层，不是 UVA-inspired 简化模型。
运行时直接实例化已安装仓库中的
`unified_video_action.policy.unified_video_action_policy.UnifiedVideoActionPolicy`，因此实际使用的仍是：

- 原 `AutoencoderKL` 和 `kl16.ckpt`；
- 原 `MAR` encoder/decoder、masking 和 `sample_tokens()`；
- 原 `GaussianDiffusion`、`DiffLoss`、`DiffActLoss`；
- 原 `process_data()`、`get_vae_latent()` 和 `get_trajectory()`；
- 原任务 Hydra 配置和原 checkpoint 加载逻辑。

策略包只有 LeRobot 配置、数据字段适配、policy wrapper 和 processor 四个源码文件：

```text
lerobot_policy_uva/
├── pyproject.toml
└── src/
    └── lerobot_policy_uva/
        ├── __init__.py
        ├── configuration_uva.py
        ├── modeling_uva.py
        └── processor_uva.py
```

## 安装

必须在同一个 Python 3.12 环境中依次安装原 UVA 和本接入包：

```bash
cd D:/UVA/unified_video_action
pip install -e .

cd D:/UVA/unified_video_action/lerobot_policy_uva
pip install -e .
```

原仓库的 `setup.py` 只注册包名，没有声明完整依赖。首次建环境时仍需安装其运行依赖；
本策略包的 `pyproject.toml` 已声明 MAR/LeRobot 接入路径直接需要的可解析依赖。

确认两个包来自预期位置：

```bash
python -c "import unified_video_action, lerobot_policy_uva; print(unified_video_action.__path__); print(lerobot_policy_uva.__path__)"
```

## 必需的预训练资产

默认严格检查原 UVA 资产，不存在时立即报错，不会退化成随机 VAE：

```bash
--policy.legacy_vae_checkpoint=D:/checkpoints/kl16.ckpt
--policy.legacy_mar_checkpoint=D:/checkpoints/checkpoint-last.pth
```

`legacy_mar_checkpoint` 可以是原始 MAR checkpoint，也可以是 UVA 第一阶段保存的 video checkpoint；
加载仍由原 `UnifiedVideoActionPolicy.load_pretrained_model()` 完成。

## LeRobotDataset 时间窗口

原 UVA 训练窗口固定为 32 步，并设置 `pad_before=1, pad_after=7`。LeRobotDataset
当前采样帧对应原序列第 1 槽，因此时间索引是：

```text
observation delta: -1, 0, 1, ..., 30  # 共 32 帧
action delta:      -1, 0, 1, ..., 30  # 共 32 步
```

原 `get_vae_latent()` 将 observation 切成 16 帧条件和 16 帧未来监督；原
`get_trajectory()` 再选动作窗口的第 15～30 槽作为 16 步目标。配置同时设置
`drop_n_last_frames=23`，与 `-1..30` 一起复现原 `horizon=32, pad_before=1,
pad_after=7` 的 episode 首尾采样范围。不要覆盖
这些设置，也不要把窗口缩成简化模型使用的 4 帧。

## 原 UVA 两阶段训练

以 PushT 为例。第一阶段训练原视频生成目标：

```bash
lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --policy.type=uva \
  --policy.legacy_config_name=uva_pusht \
  --policy.legacy_stage=video \
  --policy.legacy_vae_checkpoint=D:/checkpoints/kl16.ckpt \
  --policy.legacy_mar_checkpoint=D:/checkpoints/mar_base/checkpoint-last.pth \
  --policy.device=cuda \
  --batch_size=32 \
  --steps=200000 \
  --output_dir=outputs/uva_pusht_video
```

第二阶段加载第一阶段权重，并启用原 `DiffActLoss`：

```bash
lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --policy.type=uva \
  --policy.pretrained_path=outputs/uva_pusht_video/checkpoints/last/pretrained_model \
  --policy.legacy_config_name=uva_pusht \
  --policy.legacy_stage=action \
  --policy.legacy_vae_checkpoint=D:/checkpoints/kl16.ckpt \
  --policy.device=cuda \
  --batch_size=32 \
  --steps=200000 \
  --output_dir=outputs/uva_pusht_action
```

注意：`legacy_mar_checkpoint` 只接受原 UVA `.pth/.ckpt` 格式。LeRobot 第一阶段输出的是
完整 wrapper checkpoint，因此第二阶段通过 `--policy.pretrained_path=...` 加载。接入层先按
`legacy_stage=action` 构造含原 `DiffActLoss` 的模型，再由 LeRobot 以 `strict=False` 加载第一阶段
权重；新增 action head 保持初始化状态。这对应原脚本“加载 video 权重后新增动作扩散头”的流程。

可继续传入原 Hydra override：

```bash
'--policy.legacy_config_overrides=["model.policy.autoregressive_model_params.cfg=1.5"]'
```

## 数据 feature 要求

LeRobotDataset 可以负责存储和 32 步采样，但 feature 的物理语义必须与原 UVA 一致：

| 配置 | 图像 | state 语义 | action |
|---|---|---|---:|
| `uva_pusht` | 1 个主相机 | `agent_pos`，2D | 2D |
| `uva_libero10` | agent view | 原 UVA LIBERO 不使用 proprio | 10D |
| `uva_toolhang` | side + wrist | eef pos 3 + quat 4 + gripper 2 | 10D |
| `uva_robocasa` | left + wrist + right | ee pos 3 + ee ori 3 + gripper 2 + joints 7 | 12D |
| `uva_umi` | camera0 | eef pos 3 + rotation6D 6 + gripper 1 + rotation-to-start6D 6 | 10D |

低维字段可以作为独立 LeRobot features 保存，也可以按表中顺序拼接为
`observation.state`。相机名称无法自动判断时，显式设置：

```bash
--policy.primary_image_key=observation.images.front \
--policy.wrist_image_key=observation.images.wrist \
--policy.right_image_key=observation.images.right
```

非 UMI 任务由 LeRobot processor 使用 dataset min/max 映射到 `[-1,1]`，与原
`LinearNormalizer(mode="limits")` 相同；UMI 保持 identity。原 policy 内部设置
`normalizer_type=none`，因此不会二次归一化。

## Rollout 和 benchmark

动作阶段 checkpoint 可以直接走 LeRobot rollout/eval：

```bash
lerobot-eval \
  --policy.path=outputs/uva_pusht_action/checkpoints/last/pretrained_model \
  --env.type=pusht \
  --eval.n_episodes=50
```

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/uva_real/checkpoints/last/pretrained_model \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --task="pick up the cube"
```

`select_action()` 保留 16 帧历史，调用原 UVA `predict_action()` 和
`MAR.sample_tokens()`，再执行前 `n_action_steps=8` 个动作。

### RoboCasa 的严格限制

当前 LeRobot RoboCasa benchmark 暴露的标准 `observation.state` 是 16D，字段含义与原 UVA
训练使用的 15D `ee_pos + ee_ori + gripper_states + joint_states` 不同。本适配器会明确报错，
不会切掉一维后冒充兼容。要忠实运行 UVA RoboCasa，必须让 dataset 和 env processor 输出原
15D 表示或四个原字段；否则只能说环境接口能运行，不能说 benchmark 输入与原 UVA 等价。

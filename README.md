<h1 align="center">RoboICL: Embodied In-Context Learning with GPT-6 Astra</h1>

<p align="center"><em>Multimodal Models as Few-Shot Robot Learners</em></p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.34261"><img src="https://img.shields.io/badge/arXiv-2609.34261-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv paper"></a>
  <a href="https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/"><img src="https://img.shields.io/badge/Project%20page-RoboICL-f15b2b" alt="Project page"></a>
  <a href="https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/#results-and-episodes"><img src="https://img.shields.io/badge/Episodes-929%20rollouts-2f7a47" alt="Interactive evaluation episodes"></a>
</p>

<p align="center">
  <a href="https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/">
    <img src="images/teaser_benchmark.png" width="100%" alt="RoboICL category-level performance on the RoboDojo benchmark">
  </a>
</p>

<p align="center"><em>RoboICL uses interaction memory alone on Open and one demonstration on the other categories.</em></p>

## News

- **2026-10-07** For reproducible rollouts, we recommend the official API or a stable third-party relay, such as DMXAPI's `gpt-6-astra-ssvip`, rather than an unstable coding-plan endpoint.
- **2026-09-29** Paper released on arXiv, together with the code and 800+ interactive evaluation rollouts.
- **2026-09-19** Research Preview launched.

## Overview

RoboICL is an in-context robot-control framework that adapts a frozen
general-purpose vision-language model from demonstrations and its own execution
history. It requires no robot-specific parameter update and no learned
vision-language-action model.

RoboICL separates context into two complementary sources:

- **Demonstration context** provides recorded robot trajectories as executable
  few-shot examples.
- **Interaction memory** retains the model's own actions, controller feedback,
  and resulting observations during the current rollout.
- **A shared interaction grammar** presents both sources as
  `observation -> action -> receipt -> observation` records.
- **Bounded anchored memory** preserves selected early and intermediate
  interactions while retaining the latest result for immediate correction.

The released runner connects GPT-6 Astra to RoboDojo through one validated
action interface. At each request, the model observes a triptych of the left
wrist, head, and right wrist cameras together with robot proprioception, then
predicts a bounded sequence of dual-arm Cartesian actions for closed-loop
execution.


## Installation

### Prerequisites

- Linux with an NVIDIA GPU and a working Vulkan/CUDA stack
- Git and Git LFS
- an Isaac Sim 6.0.1 simulator environment (Python 3.12, Isaac Lab 3.0)
- a separate Python environment for the multimodal policy process

The simulator and policy processes intentionally use separate interpreters
because their `websockets` and simulator dependency constraints differ.

### Clone the repository

```bash
git clone https://github.com/Mosi-AI/RoboICL.git
cd RoboICL
```

RoboDojo, XPolicyLab, and IsaacLab sources are tracked directly under `third_party/`.
cuRobo is installed in the simulator environment from the upstream commit pinned
in `configs/upstream.lock.json` and `setup/requirements-sim.txt`.
There are no Git submodules or nested dependency checkouts. Upstream revisions
and the required simulator version are recorded in `configs/upstream.lock.json`;
source entry points, interpreter metadata, and the installed cuRobo commit are
checked before launch.
Third-party licenses remain alongside their sources. See
[`third_party/README.md`](third_party/README.md) for the import layout and vendoring changes.

### Install the environments once

Run from the repository root. The two environments have distinct jobs:

| Environment | Purpose |
|---|---|
| `roboicl-sim` | Isaac Sim 6.0.1, Isaac Lab, cuRobo, and robot simulation |
| `roboicl-policy` | Model API requests and the policy server |

Create fresh environments to keep other projects' dependencies and source paths
out of this installation. Skip this installation section if they are already configured.

**Install the simulator:**

```bash
conda env create -f setup/environment-sim.yml
conda activate roboicl-sim
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  -r setup/requirements-sim.txt -c setup/constraints-sim.txt \
  --extra-index-url https://pypi.nvidia.com
python -m pip check
```

This installs Isaac Sim `6.0.1.0`, cuRobo at the pinned upstream commit, and the
Isaac Lab packages and Kit visualizer from this repository. `pip check` should report
`No broken requirements found.`

**Install the policy process:**

```bash
conda env create -f setup/environment-policy.yml
conda activate roboicl-policy
python -m pip install -r setup/requirements-policy.txt
python -m pip check
```

### Prepare each terminal before running

From the repository root, only these two commands are needed:

```bash
conda activate roboicl-policy
source setup/paths.sh
```

`setup/paths.sh` selects both environments' Python interpreters and the project's
`data/` and `results/` directories. It also disables user-site packages and sets
up the simulator's environment library when available. The launcher starts the
simulator with `roboicl-sim` automatically; keep `roboicl-policy` active for the
commands below. Nothing needs to be saved with `conda env config vars set`.

For custom environment locations or data directories, set the corresponding
`ROBOICL_*` variables before sourcing the script; existing overrides are preserved.

## Data preparation

### RoboDojo assets

Fetch the complete frozen RoboDojo asset tree directly into `data/Assets`.
The script checks out the revision in `configs/data.lock.json` and verifies the
download. `data/Assets` is a real directory. Git/LFS staging files are temporary
and are removed after installation; no separate asset cache or symbolic link is needed.

`data/` contains the asset files, installation records, and task demonstrations
under `data/runtime-data/`.

```bash
"$ROBOICL_POLICY_PYTHON" setup/fetch_assets.py \
  --data-root "$ROBOICL_DATA_ROOT"
```

### One-shot references

The paper benchmark contains exactly 8 Open tasks evaluated zero-shot and 22
non-Open tasks supplied with one demonstration. The complete source-episode
identity, HDF5 SHA256, prediction horizon, and all 12 selected action-window
starts are archived in
[`configs/references/one_shot_j12_b12.json`](configs/references/one_shot_j12_b12.json).

Fetch and deterministically build any archived one-shot reference, for example:

```bash
"$ROBOICL_POLICY_PYTHON" setup/fetch_reference.py deposit_coin \
  --data-root "$ROBOICL_DATA_ROOT"

"$ROBOICL_POLICY_PYTHON" setup/fetch_reference.py insert_key \
  --data-root "$ROBOICL_DATA_ROOT"
```

The source is checked by SHA256, and the builder uses the exact archived starts
rather than silently re-running a heuristic. The generated bundle contains 12
non-overlapping action blocks and their starting/result observations.

TRAIN and LIVE use different selectors. TRAIN uses the exact action-window
starts recorded in the setting manifest. LIVE does not sample video frames:
eleven uniformly spaced time anchors retain the first complete executed
interaction that crosses each anchor, while the twelfth slot retains the latest
complete interaction.

To build a reference for another task from a local HDF5 episode:

```bash
"$ROBOICL_POLICY_PYTHON" -m roboicl.reference build \
  --task "$TASK" \
  --source "$ROBOICL_DATA_ROOT/runtime-data/$TASK/train/episode_0000000.hdf5" \
  --source-id "runtime-data/$TASK/train/episode_0000000.hdf5" \
  --source-sha256 "$SOURCE_SHA256" \
  --output "$ROBOICL_DATA_ROOT/runtime-data/$TASK/reference_1shot_train12_endpoint_h$H" \
  --horizon "$H" \
  --chunks 12 \
  --selected-starts "$START_0,$START_1,...,$START_11"
```

## Run RoboICL

The release provides two main protocols:

| Protocol | Demonstrations | TRAIN blocks `J` | LIVE memory blocks `B` |
|---|---:|---:|---:|
| `configs/protocols/zero_shot_b25.json` | 0 | 0 | 25 |
| `configs/protocols/one_shot_j12_b12.json` | 1 | 12 | 12 |

Both use the task-adaptive action horizons in
`configs/adaptive_horizons.json`.

### 1. Preflight without an API request

```bash
"$ROBOICL_POLICY_PYTHON" -m roboicl.run deposit_coin \
  --shots 0 \
  --seed 0 \
  --layout 0 \
  --profile configs/protocols/zero_shot_b25.json \
  --dry-run
```

This validates the task, layout, assets, profile, interpreters, and bundled
sources, installed cuRobo provenance, and Isaac Sim 6.0.1.0 / Python 3.12,
without starting the simulator or calling the model API.

### 2. Validate the simulator and cameras

```bash
"$ROBOICL_POLICY_PYTHON" -m roboicl.run deposit_coin \
  --shots 0 \
  --seed 0 \
  --layout 0 \
  --gpu 0 \
  --profile configs/protocols/zero_shot_b25.json \
  --capture-only
```

`--capture-only` launches the scene and verifies the three RGB streams without
making a model request or executing an action.

A successful run reports `capture_complete_no_api` and writes
`capture/capture_probe.json` plus sample PNGs in the printed run directory.
The adapter enables the Kit visualizer in headless mode so Isaac Lab 3 continues
to update camera render products. On failure, inspect that run's `client.log`
for the simulator traceback; this check does not require an API key.

RoboDojo layouts, policy observations, and cuRobo poses retain WXYZ quaternion
order. Robot initialization converts to Isaac Lab 3's XYZW order, and link-state
reads convert back at the robot manager boundary. This preserves the intended
robot mounting orientation and keeps observed poses aligned with IK targets.

The compatibility layer also corrects the deprecated Warp prim utility's
world-to-local conversion to apply the inverse parent transform. This preserves
camera poses when prim wrappers rebuild transform properties, including mounted
wrist cameras, without changing the camera configuration.

### 3. Run a scored zero-shot rollout

```bash
export ASTRA_API_KEY=...

"$ROBOICL_POLICY_PYTHON" -m roboicl.run deposit_coin \
  --shots 0 \
  --seed 0 \
  --layout 0 \
  --gpu 0 \
  --profile configs/protocols/zero_shot_b25.json
```

### 4. Run a scored one-shot rollout

```bash
export ASTRA_API_KEY=...

"$ROBOICL_POLICY_PYTHON" -m roboicl.run deposit_coin \
  --shots 1 \
  --seed 0 \
  --layout 0 \
  --gpu 0 \
  --profile configs/protocols/one_shot_j12_b12.json
```

Replace the task, seed, and layout as needed. Use `--reference /path/to/bundle`
for a one-shot task without a published reference path. A process exit code of
zero means evaluation completed; task success and progress are reported by the
native RoboDojo evaluator in the generated `result.json`.

### Use a local vLLM model

The policy supports `api_mode=chat_completions` as well as the existing Responses
interface. The local profile targets the running `Qwen3.8-27B` service:

```bash
export ASTRA_API_KEY=local-vllm

"$ROBOICL_POLICY_PYTHON" -m roboicl.run deposit_coin \
  --shots 0 --seed 0 --layout 0 --gpu 0 \
  --profile configs/protocols/local_qwen_zero_shot.json
```

The key is a placeholder for an unauthenticated local server; use its actual key
if server authentication is enabled. Edit the profile's `model` and `endpoint`
when using a different service. This profile enables Qwen thinking through
`chat_template_kwargs`, allows 16384 output tokens for reasoning and the action
tool call together, and uses non-streaming
Chat Completions. The adapter converts camera images, action tool calls, and
execution receipts while preserving tool-call IDs. Responses-only reasoning
and cache fields are omitted; truncated model outputs are rejected before any
action is executed. Chat streaming is currently unsupported.

The verified local `vLLM 0.24.0+cu129` launch uses
`CUDA_HOME=/usr/local/cuda-12.9` for the SM120 FlashInfer kernels, with
`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`. It omits
`--speculative-config`: the earlier MTP launch returned HTTP 500 for thinking
with strict action tools. Thinking and the strict `act` tool have been verified
with a recorded first-frame request under the working launch. This verifies the
interface; action bounds and physical task success still require separate checks.

The server must support image inputs and function tools. Its context window and
GPU allocation must also accommodate the policy history alongside Isaac Sim.
Interface checks alone do not establish task success; inspect the evaluator's
`result.json` after a rollout.

Add `--gui` to show the Isaac Sim window during evaluation (requires a graphical
desktop / working `DISPLAY`):

```bash
"$ROBOICL_POLICY_PYTHON" -m roboicl.run deposit_coin \
  --shots 0 --seed 0 --layout 0 --gpu 0 \
  --profile configs/protocols/local_qwen_zero_shot.json --gui
```

Use `--gui --capture-only` for a short scene/camera preview without model requests
or robot actions; the window closes when the camera probe finishes. Evaluation
remains headless by default. The GUI viewport and the three RGB observations
sent to the policy are separate views.

### Local Qwen with one demonstration (ICL)

Use `local_qwen_one_shot.json` to inject one recorded demonstration into the
local Qwen context. It keeps thinking enabled and provides 12 TRAIN action
blocks, their observations and results, plus 12 retained LIVE memory blocks.
The model weights are unchanged. The zero-shot profile requires `--shots 0`
and cannot be used with `--shots 1`.

Prepare the official, content-verified `deposit_coin` demonstration once:

```bash
"$ROBOICL_POLICY_PYTHON" setup/fetch_reference.py deposit_coin \
  --data-root "$ROBOICL_DATA_ROOT"
```

The data lives under `$ROBOICL_DATA_ROOT` (default: this checkout's `data/`):

```text
runtime-data/deposit_coin/
├── train/episode_0000000.hdf5                  # original 191-frame episode
└── reference_1shot_train12_endpoint_h5/
    ├── train_reference_bundle.json            # 12 five-step action blocks and observations
    └── verification.json                     # source and bundle verification
```

The source episode is approximately 172 MB. Its SHA256 and the 12 selected
window starts are locked in `configs/references/one_shot_j12_b12.json`.
The runner automatically loads this reference when `--shots 1` is selected;
the protocol JSON itself contains settings, not the demonstration data.
These files are separate from scene assets in `data/Assets` and from evaluation
outputs in `results/`.

For the locally prepared `deposit_coin` data, open
`data/runtime-data/deposit_coin/reference_1shot_train12_endpoint_h5/preview.md`
to view the selected camera frames, chunk indices, and the first real action
tensor. Its `preview/` images are for inspection only; the runner consumes the
verified bundle, not these previews. Camera images are embedded as data URLs in
the bundle JSON, so they do not appear as separate image files there.

Validate without launching the simulator or requesting inference:

```bash
"$ROBOICL_POLICY_PYTHON" -m roboicl.run deposit_coin \
  --shots 1 --seed 0 --layout 0 --gpu 0 \
  --profile configs/protocols/local_qwen_one_shot.json --dry-run
```

Run with the demonstration and GUI:

```bash
export ASTRA_API_KEY=local-vllm

"$ROBOICL_POLICY_PYTHON" -m roboicl.run deposit_coin \
  --shots 1 --seed 0 --layout 0 --gpu 0 \
  --profile configs/protocols/local_qwen_one_shot.json --gui
```


### 下载全部任务示范与更换任务

当前代码配置了 **42 个任务**。官方固定版本的数据集发布了其中 **34 个任务**的原始示范；8 个 Open 类任务采用零样本设置，没有发布对应示范。论文协议锁定了 22 个任务的 one-shot reference。

下载每个有公开数据的任务的一条示范，并构建可校验的锁定 ICL reference：

```bash
"$ROBOICL_POLICY_PYTHON" setup/fetch_task_data.py \
  --data-root "$ROBOICL_DATA_ROOT" --workers 3
```

此命令准备每任务一条用于查看或 ICL 的经验数据，不下载每个任务的全部数百条轨迹；已有正确文件会复用，部分下载可续传。普通下载遇到 403 时，会尝试 PATH 中的官方 `hf` 客户端，也可用 `--hf-cli /path/to/hf` 指定它。所有源文件都按固定版本与 SHA256 校验。下载清单及逐任务结果保存在：

```text
$ROBOICL_DATA_ROOT/runtime-data/task_data_inventory.json
$ROBOICL_DATA_ROOT/runtime-data/README.md
$ROBOICL_DATA_ROOT/runtime-data/<任务名>/train/episode_XXXXXXX.hdf5
$ROBOICL_DATA_ROOT/runtime-data/<任务名>/reference_1shot_train12_endpoint_h<H>/
```

`runtime-data/README.md` 是可点击的数据索引，列出每个任务的原始 HDF5、
ICL bundle 和实际准备状态；JSON 清单另外记录源版本、文件大小与 SHA256。

`H` 是任务自动选择的动作步数，不要把所有任务的 reference 都写成 `h5`。`swap_blocks` 和 `play_Xylophone` 的锁定示范使用 episode 1；其余锁定示范使用 episode 0。原始示范存在不等于已有锁定 ICL reference：只有清单状态为 `reference_ready` 的任务可以直接使用下方 one-shot 命令。若远端文件不符合锁定身份，工具会保留错误记录，不替换校验哈希。

`build_tower` 和 `fill_pen_holder` 的历史锁定源对象目前不可下载，官方 LFS 返回对象不存在。这两个任务下载当前固定公开版本的原始示范，单独存入 `runtime-data/<任务名>/public/train/`，清单状态为 `source_verified_reference_unavailable`，使用 `--shots 0`。`organize_table` 使用已确认与原 SHA256 一致的历史版本，可正常构建 ICL reference。原论文的 22 条 reference 定义仍保留；目前可准备的是其中 20 条。

**更换任务只需修改位置参数或 `TASK`。** 例如把投币换成插钥匙：

```bash
TASK=insert_key
export ASTRA_API_KEY=local-vllm

"$ROBOICL_POLICY_PYTHON" -m roboicl.run "$TASK" \
  --shots 1 --seed 0 --layout 0 --gpu 0 \
  --profile configs/protocols/local_qwen_one_shot.json --gui
```

先检查任务、资产与示范是否齐全，可将上述命令末尾的 `--gui` 换成 `--dry-run`，不启动仿真也不请求模型。

零样本运行适用于全部已配置任务；只需同时更换 shots 和 profile：

```bash
TASK=stack_blocks
export ASTRA_API_KEY=local-vllm

"$ROBOICL_POLICY_PYTHON" -m roboicl.run "$TASK" \
  --shots 0 --seed 0 --layout 0 --gpu 0 \
  --profile configs/protocols/local_qwen_zero_shot.json --gui
```

任务名称区分大小写：保留 `play_Xylophone`、`push_T`、`swap_T` 的原始大小写。`--seed` 和 `--layout` 选择场景实例；控制步数、任务说明和默认示范路径随任务自动加载。切换任务不会重新训练模型。

下面的“协议”是公开示范覆盖范围，不是任务成功率。20 个 reference 可用任务在校验通过后使用 `--shots 1`；另外 12 个任务仅下载原始示范，使用 `--shots 0`；两条历史源不可用的任务也使用 `--shots 0`。

| 任务名（命令参数） | 任务内容 | 示范与运行协议 |
|---|---|---|
| `align_blocks` | 用三角尺推动方块，使其排成直线 | Open：`--shots 0`，无公开示范 |
| `arrange_largest_number` | 排列数字牌，组成最大数 | 原始示范：`--shots 0` |
| `build_tower` | 用木块和木板搭建稳定高塔 | 公开原始示范：`--shots 0`，历史 ICL 源不可用 |
| `classify_objects` | 按类别将物体放入不同篮子 | 锁定 ICL：`--shots 1` |
| `classify_objects_by_language` | 按语言要求将物体分类到指定篮子 | Open：`--shots 0`，无公开示范 |
| `cover_blocks` | 覆盖方块，记住颜色后按指定顺序揭开 | 锁定 ICL：`--shots 1` |
| `deposit_coin` | 夹起硬币并插入存钱罐 | 锁定 ICL：`--shots 1` |
| `fasten_screws` | 按颜色匹配螺丝与螺母并拧紧 | 锁定 ICL：`--shots 1` |
| `fill_egg_holder` | 将四个鸡蛋放入蛋盒并合盖 | 锁定 ICL：`--shots 1` |
| `fill_pen_holder` | 一手持笔筒，另一手逐支装笔并放回 | 公开原始示范：`--shots 0`，历史 ICL 源不可用 |
| `fold_clothes` | 折叠衣物 | 原始示范：`--shots 0` |
| `general_pickup` | 按语言要求识别并拾取目标物体 | Open：`--shots 0`，无公开示范 |
| `hang_mugs` | 将三个杯子挂到杯架上 | 原始示范：`--shots 0` |
| `imitate_sorting_sequence` | 记住对面机器人的顺序并模仿分类 | 锁定 ICL：`--shots 1` |
| `insert_key` | 抓取并交接钥匙，插入锁孔后转动 | 锁定 ICL：`--shots 1` |
| `insert_tubes` | 将三根试管依次插入试管架 | 锁定 ICL：`--shots 1` |
| `make_kong` | 观察麻将弃牌，选择对应牌完成杠 | 锁定 ICL：`--shots 1` |
| `make_toast` | 将两片面包放入烤面包机并按下拨杆 | 原始示范：`--shots 0` |
| `match_and_pick_from_conveyor` | 记住传送带首个物体并拾取后续匹配物体 | 锁定 ICL：`--shots 1` |
| `organize_table` | 整理鼠标、键盘、摆件和杂物到指定位置 | 锁定 ICL：`--shots 1` |
| `pack_objects_into_box` | 将物体按规定朝向装入盒子 | 原始示范：`--shots 0` |
| `pick_from_conveyor_by_image` | 根据目标图片，从传送带拾取物体放入篮子 | Open：`--shots 0`，无公开示范 |
| `play_Xylophone` | 抓起槌子，从左到右敲击木琴 | 锁定 ICL：`--shots 1` |
| `play_stacking_toy` | 把不同形状的玩具块放到匹配立柱上 | 锁定 ICL：`--shots 1` |
| `play_tic_tac_toe` | 与对手轮流下井字棋 | 锁定 ICL：`--shots 1` |
| `plug_in_charger` | 将充电插头插入插线板 | 锁定 ICL：`--shots 1` |
| `pour_balls_into_vase` | 把杯中的小球倒入花瓶 | 锁定 ICL：`--shots 1` |
| `pour_by_language` | 按语言指定的瓶子和碗倒液体 | Open：`--shots 0`，无公开示范 |
| `pour_liquid_into_cup` | 将瓶中液体倒入杯子 | 原始示范：`--shots 0` |
| `press_by_number` | 按数字卡次数按按钮并确认 | 锁定 ICL：`--shots 1` |
| `push_T` | 推动 T 形块，与目标轮廓精确对齐 | 原始示范：`--shots 0` |
| `put_bottles_into_dustbin` | 拾取并交接瓶子，投入垃圾桶 | 锁定 ICL：`--shots 1` |
| `solve_equation` | 选择缺少的数字或运算符，补全算式 | Open：`--shots 0`，无公开示范 |
| `sort_nesting_dolls_by_size` | 按大小将套娃从小到大排列 | 原始示范：`--shots 0` |
| `stack_blocks` | 将三个方块叠成稳定堆叠 | 原始示范：`--shots 0` |
| `stack_blocks_by_language` | 按语言指定的颜色顺序叠方块 | Open：`--shots 0`，无公开示范 |
| `stack_bowls` | 将三个碗叠放在一起 | 原始示范：`--shots 0` |
| `store_laptop_and_headphones` | 收好耳机，合上笔记本电脑并竖放 | 原始示范：`--shots 0` |
| `store_tools_in_toolbox` | 把工具放入工具箱的对应槽位 | Open：`--shots 0`，无公开示范 |
| `swap_T` | 双手交换两个 T 形块的位置与姿态 | 锁定 ICL：`--shots 1` |
| `swap_blocks` | 借助空垫交换两个方块，每次移动后按按钮 | 锁定 ICL：`--shots 1` |
| `sweep_blocks` | 双手配合使用扫帚和簸箕收集方块 | 原始示范：`--shots 0` |



## Repository layout

```text
roboicl/
├── run.py                 preflight and process orchestration
├── robodojo.py            single-layout RoboDojo adapter
├── rollout.py             observation/action execution loop
├── policy_server.py       XPolicyLab WebSocket policy entrypoint
├── reference.py           reference builder and verifier
└── policy/                prompting, memory, validation, and transport
configs/
├── protocols/             zero-shot and one-shot protocol profiles
├── tasks/                 official documentation for 42 tasks
├── adaptive_horizons.json task-specific action horizons
├── data.lock.json         frozen asset revision
├── references/             setting-level source episodes and exact TRAIN windows
└── upstream.lock.json     bundled source provenance and simulator version
setup/
├── fetch_assets.py
├── fetch_reference.py
├── fetch_task_data.py      verified per-task downloads and locked references
├── requirements-policy.txt
├── requirements-sim.txt
└── paths.sh
third_party/
├── RoboDojo/
├── XPolicyLab/
└── IsaacLab/
```

<p align="center">
  <img src="images/wechat.jpg" width="330" alt="RoboICL WeChat discussion group QR code">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="images/rednote.png" width="330" alt="RoboICL RedNote discussion group QR code">
</p>

## Citation

```bibtex
@article{liu2026roboicl,
  title={RoboICL: Embodied In-Context Learning with GPT-6 Astra},
  author={Liu, Fangcheng and Shen, Yeqing and Cheng, Anda and Mi, Weishi and Tang, Chao and Liu, Chenyuan and Xiang, Yushun and Li, Tingguang and Li, Yong-Lu and Tang, Yehui},
  journal={arXiv preprint arXiv:2609.34261},
  year={2026}
}
```

## License

RoboICL's project-owned code is released under the [MIT License](LICENSE).
Third-party sources and benchmark assets retain their original licenses and
terms.

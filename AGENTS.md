# AGENTS.md

> AI 编程助手与人类开发者的共同开发规范，由 pi-init 生成，请结合实际修订。

## 1. 项目概述 (Project overview)

RSC 是一条为 Roblox 做的正弦/和声合成音频压缩线：音频进去，另一端出来不那么干净。
RSC2 是当前上线的版本（additive synthesis，v3 bitstream，Roblox 端 Lua 解码器已发布）；
RSC3 是后继者——块变换、快速 MDCT、均匀 scalefactor band、反向 waterfill 按心理声学
threshold 分配比特，已有容器 / 编码器 / 解码器，但解码器还是 Python，所以还没上 Roblox。

- 语言：Python 3.12（`.python-version`），Roblox 侧 Luau
- 数值：`numpy`、`scipy`、`librosa`、`tqdm`；`numba`（解码器的位级步进走 JIT，见 `rsc3_bits.py`）、`sounddevice`（监听播放）
- 工程：扁平脚本集合，**不是可安装包**（`pyproject.toml` 里 `[tool.uv] package = false`），没有构建步骤
- 无 Web 框架、无 API Key、无数据库；仓库自己声明 "Vibe Coded"，README 的随意语气不代表代码可以随意

## 2. 快速命令 (Quick commands)

```bash
# 安装依赖（Python 版本由 .python-version 决定）
uv sync
# 或者传统安装（requirements.txt 只有 tqdm 钉了版本）
pip install -r requirements.txt

# RSC2 编码器 / 解码器（这条线是上线的那条）
python rsc_encoder.py --input audio.wav --output audio.rsc
python rsc_encoder.py --input audio.wav --output audio.rsc --partials 12 -r 22050
python rsc_decoder.py --input audio.rsc --output decoded.wav
python rsc_decoder.py --input audio.rsc --output decoded.wav --no-lut
# 注意两个 CLI 的 --samplerate 都只允许 22050 / 44100，argparse 里就钉死了

# RSC3：从 rsc3_prototype/ 里跑（README 的要求，产出路径按 cwd 解析）
cd rsc3_prototype
uv run python rsc3_encode.py input.wav --bitrate 24 --pvq --fill --ms
uv run python rsc3_encode.py input.wav --budget 32     # 跳过 rate search，直接给平面数
uv run python rsc3_decode.py input.rsc3 -o decoded.wav

# 测试 / 校验（都是普通脚本，没有 pytest 配置；退出码 0 = 通过）
uv run python test_stream.py        # 语法 + 声音一致性回归（对比 _ref_stream.py）
uv run python test_roundtrip.py     # encode -> file -> decode，两个 CLI 不许不一致
uv run python transform_check.py    # 变换自己的证明：kernel 匹配 + 全信号 OLA
uv run python perf_check.py         # 改了性能/语法之后必跑；--bless 重建基线

# 想验证 "编译过的路径 = 同一套语法"，用这个开关，别改代码：
RSC3_NO_JIT=1 uv run python rsc3_decode.py input.rsc3 -o a.wav   # 纯 Python 步进
uv run python rsc3_decode.py input.rsc3 -o b.wav                 # JIT 步进，两遍应当逐样本相同

# 度量
uv run python gainfit_es.py         # 拟合增益后的 ES（两个数都打）
uv run python rate_curve.py         # kbit/s vs quality，每个字段都记账
uv run python _decode_speed.py      # 解码侧分段计时

# 不在 rsc3_prototype/ 里时，上面换带路径的形式等价：
uv run python rsc3_prototype/test_stream.py

# Lint / 格式化：项目里没有任何配置 —— 待补充
# 想要就先跑一次性检查，别在没配置前顺手改一堆文件的格式：
uvx ruff check . && uvx ruff format --diff .
```

依赖的真实入口是 `--help`：CLI 跑完会自己打印配套的解码命令。

## 3. 目录结构说明 (Structure)

```
/                          仓库根 = RSC2 那条线
├── rsc_encoder.py         RSC2 编码器 CLI（argparse，文档头里有真实命令）
├── rsc_decoder.py         RSC2 Python 解码器 CLI
├── rsc_decoder.lua        Roblox 运行时解码器（已发布，open-source）
├── rsc_serverfetch.lua    Roblox 服务端取流
├── RSCv6.rbxl             Roblox place 本体
├── noisegen.py            生成噪声 fixture
├── refsine.py             生成 sine_<freq>hz_<n>s.wav 测试音
├── testing.py             交互式监听 / 播放对比（需要 sounddevice）
├── pyproject.toml         依赖 + [tool.uv] package=false
└── uv.lock                明确不 ignore，改了依赖必须提交

rsc3_prototype/            后继版本，所有度量都从这里跑
├── rsc3_stream.py         容器 + bit 语法 + 解码器读的那一帧（改动最频繁）
├── rsc3_encode.py         编码器 CLI：--bitrate 搜 budget，--budget 直接花
├── rsc3_decode.py         解码器 CLI，只依赖文件本身
├── rsc3_codec.py          模型内核：变换、threshold、分配、量化
├── rsc3_progress.py       终端进度条
├── test_stream.py         语法回归；缺 rsc2_testing_lab fixture 时打 SKIP 而不是崩
├── test_roundtrip.py      encode -> file -> decode 的端到端一致性
├── perf_check.py          六个配置：payload 逐字节 + 解码样音逐样本地比
├── rate_curve.py / audible_sweep.py / hf_audit.py / gainfit_es.py    度量工具
├── ab_codec.py / _make_listening.py / _listen_shape.py               听音对照产物生成
├── transform_check.py     变换正确性
├── _ref_codec.py / _ref_stream.py    钉住的旧语法，用来证明"是优化不是新 codec"
└── _*.py / *_experiment.py / vector_hits.py    一次性探针或已定论的实验

rsc2_testing_lab/          对照实验与 RSC2 参考实现
└── 注意：cam.wav / ij.wav / cam15.wav 不在本 checkout 里，
    依赖它们的测试会 SKIP。这不代表测试通过。
```

## 4. 代码风格与规范 (Code style)

没有 linter/formatter 配置，风格靠现有代码，下面这些是真的在用：

- 模块和函数的 docstring 写"为什么"，不是"是什么"：写清楚这条改动值多少钱、用什么量出来的、
  为什么不能用别的做法。改行为时要顺手更新它，否则文档会先变成假话
- 数值路径向量化优先，用 numpy 批量而不是 Python 内层循环（性能记录就是这么来的：
  waterfill 从 60 遍扫描改成 `argpartition` 选择，约 5x，且逐 bit 相同）
- 脚本一律 `if __name__ == "__main__"` + `argparse`，并用
  `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` 让自己能从任何目录运行
- 路径按 repo 根相对解析（`os.path.dirname(HERE)` 找 `rsc2_testing_lab/`），README 里同时出现
  Windows 和 POSIX 写法，说明两个平台都要能跑
- 命名：`_` 前缀 = 一次性探针，或者为了"那个主张还能验"而保留的老路径
  （例：`allocate(..., search=True)` 走 `_waterfill_all`）
- 头部字段一律 **derive 优先**：能由 `sr / n2 / width / budget` 推出来的东西不发；
  推不出来、猜错了会跑偏的才发（shape word 宽度就是这样进了 header）
- 数字进 README 之前要有对应的文件度量，写"约 X dB"必须说清是哪个 fixture 哪个速率

## 5. 禁止规则 (NEVER rules)

- 🚫 禁止改 `_ref_codec.py` / `_ref_stream.py`。它们是被钉住的旧语法，唯一作用是让
  "byte-identical / sound-identical"这种话有意义。改了就等于把"优化"变成"新 codec"
- 🚫 禁止在没有 `perf_check.py` 干净通过前把性能改动算数。先 `--bless` 建基线，再改，再比
- 🚫 禁止在没记录 header 变更的情况下改 bitstream 语义。语法变了 header 必须能表达出来，
  否则老文件会被解码成 desync；漂移检查救不了你（它只抓账不平，不抓读错）
- 🚫 禁止把一个花钱买 side info 的技巧算成"免费的"——README 的原话是这类开销必须记在
  budget **里面**，否则比较不算数
- 🚫 禁止提交生成的音频：`*.wav *.mp3 *.flac *.ogg *.m4a *.opus *.rsc3 *.rsc` 都在
  `.gitignore` 里，因为它们都能从源 re-generate。也别指望 `rsc2_testing_lab/*.wav` 在
- 🚫 禁止手改 `uv.lock` 内容；依赖变更用 `uv add/remove`，然后把这个明确被 un-ignore 的
  lock 一起提交
- 🚫 禁止跳过测试直接提交：改动碰到 `rsc3_stream.py` / `rsc3_codec.py` 就必须跑
  `test_stream.py` + `test_roundtrip.py`，改到解码侧还要跑 `perf_check.py`
- 🚫 禁止手碰 `.venv/` `__pycache__/` `/listening/` `*.bak`（都是产物或可再生文件）
- 🚫 禁止把手写的硬编码阈值散落在调用点；threshold/量化这类模型参数集中在
  `rsc3_codec.py`，改了要在 README 的 "What is settled" 里认账
- ⚠️ 编码器每次运行会打印 `field accounting vs bytes on disk: drift N bits`。
  `N > 7` 意味着有字段没写或者被读了两次，下面所有 rate 数字都不再值钱——先修这个
- ⚠️ 对 budget 曲线**平坦**的结果按"harness 坏了"处理，直到证明它不是
- ✅ 优化顺序先解码器：`_decode_speed.py` 的原话是解码才是要 ship 的那一侧，编码器允许慢

## 6. Git 工作流 (Git workflow)

单人仓库（`CONTRIBUTING.md`：Only I can controbute anyway），主干 `main`，remote 在 GitHub。
现状是一次提交风格："noise shaping suff"。建议从现在起：

```
perf(stream): vectorise run_fill_words across frames      # 类型(scope): 摘要
feat(encode): shape word width in header, --shape-block
docs(readme): charge the wide word inside the budget
```

- 只推 `main`；分支只在验证一个"是不是变慢了但还一样"的假设时开
- commit 之前（在 `rsc3_prototype/` 里）：
  `uv run python test_stream.py && uv run python test_roundtrip.py` 全绿，
  涉及性能再补 `perf_check.py`
- 提交信息里带上度量：改了多少 b/f、哪个 fixture、多少 kbit/s，没有就别说
- PR：单人情况下没有 PR，改动描述写在 README 或 commit body 里

## 7. 常见任务指南 (Common tasks)

**加依赖**

```bash
uv add librosa          # 加一个依赖
uv lock && uv sync      # 重新解析并安装，提交 uv.lock
```

**跑单个测试**：直接跑那个文件，pytest 没配置，也不接受 `-k`。

```bash
uv run python test_stream.py
```

**改 bit 语法**：`write_frame` / `read_frame` 是对称的一对，改哪边都要同步改另一边的
记账函数 `frame_costs`，然后跑 `perf_check.py`（它同时管 payload 和样音）；
只动一边一定会 desync。

**调试编码器**：

```bash
uv run python rsc3_encode.py x.wav --budget 32          # 去掉 rate search 这层干扰
uv run python -m cProfile -s tottime rsc3_encode.py x.wav --bitrate 24 --pvq --fill --ms
uv run python gainfit_es.py                             # naive 和 fitted 两个数都看
```

看 decode 就分段量：`_decode_speed.py` 会把 read_frame / reconstruct / synthesise 拆开，
先看它再动手。

**做听音对照**：`_make_listening.py`、`ab_codec.py`（协议产文件）、`_listen_shape.py`，
产物落在被 ignore 的 `/listening/`。

**新加 header 字段**：加常量 + `pack_header` / `unpack_header` + 一个
derive 函数（形状照 `shape_block` / `shape_block_shift` 那对做，编码端写、解码端读），
并在 `test_stream.py` 里同时测 round-trip 与"老 header 落到默认值"。

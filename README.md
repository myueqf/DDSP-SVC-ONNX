# DDSP-SVC-ONNX

这是一个基于 .NET / ONNX Runtime 的 DDSP-SVC 推理实现，提供：

- 友好的命令行推理
- 可直接从 C# 调用的库接口

当前主要面向导出的 DDSP-SVC ONNX 模型进行本地推理与集成。

### 构建和运行  
要求：
- .NET SDK 8.0 或更高版本
- 用于 CLI 输入的任意人类干声（肯定有叭？）
- 一个包含 encoder.onnx、velocity.onnx 和 svc.json 的 DDSP-SVC ONNX 导出目录
- 依赖模型：contentvec、rmvpe 以及 pc_nsf_hifigan

#### 下载模型～

[rmvpe（Yxlllc）](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe-onnx.zip)  
[声码器（OpenVPI）](https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip)

推荐的目录结构～
```text
./Model/onnx/
  encoder.onnx
  velocity.onnx
  svc.json

./Dependencies/
  contentvec/
    contentvec.onnx
  rmvpe/
    rmvpe.onnx
  pc_nsf_hifigan_44.1k_hop512_128bin_xxx/
    vocoder.yaml
    *.onnx
```

构建：
```bash
dotnet build ./src/DdspSvc.OnnxRuntime.Cli/DdspSvc.OnnxRuntime.Cli.csproj
```

#### 检查模型解析结果

不指定目录（如果你使用默认目录结构的话嘛）：

```bash
dotnet ./src/DdspSvc.OnnxRuntime.Cli/bin/Debug/net8.0/DdspSvc.OnnxRuntime.Cli.dll inspect
```

如果你使用自定义目录：

```bash
dotnet ./src/DdspSvc.OnnxRuntime.Cli/bin/Debug/net8.0/DdspSvc.OnnxRuntime.Cli.dll inspect \
  --model-root model/onnx \
  --dependencies-root Dependencies
```

`inspect` 会输出：

- 解析到的依赖目录
- `encoder / velocity / vocoder / rmvpe / contentvec` 实际路径
- 从 `svc.json` 读取元数据（如果模型目录或其上一级存在 config.yaml，再额外读取部分默认参数作为补充）
- 从 `vocoder.yaml` 读到的声码器配置

### 推理～

如果你使用默认目录：

```bash
dotnet ./src/DdspSvc.OnnxRuntime.Cli/bin/Debug/net8.0/DdspSvc.OnnxRuntime.Cli.dll render \
  input.wav output.wav
```

使用显式模型根目录：

```bash
dotnet ./src/DdspSvc.OnnxRuntime.Cli/bin/Debug/net8.0/DdspSvc.OnnxRuntime.Cli.dll render \
  --model-root /path/to/model/onnx \
  --dependencies-root /path/to/Dependencies \
  input.wav output.wav
```

也支持显式覆写单个模型路径：

```bash
dotnet ./src/DdspSvc.OnnxRuntime.Cli/bin/Debug/net8.0/DdspSvc.OnnxRuntime.Cli.dll render \
  --model-root /path/to/model/onnx \
  --dependencies-root /path/to/Dependencies \
  --encoder-path /path/to/encoder.onnx \
  --velocity-path /path/to/velocity.onnx \
  --contentvec-path /path/to/contentvec.onnx \
  --rmvpe-path /path/to/rmvpe.onnx \
  --vocoder-path /path/to/vocoder.onnx \
  input.wav output.wav
```

如果指定了某个显式路径，则该模型不再走自动解析，其他未指定模型仍会继续按默认规则自动查找。


### 库接口用法

除了 CLI，也可以直接从 C# 里调用哦：

```csharp
using DdspSvc.OnnxRuntime;
using DdspSvc.OnnxRuntime.Hosting;
using DdspSvc.OnnxRuntime.Models;

using var runtime = SvcRuntime.Create(new SvcRuntimeOptions {
    ModelRoot = "/path/to/ddspmodel/onnx",
    DependenciesRoot = "/path/to/Dependencies",
});

runtime.ValidateRequest(new SvcInferenceRequest {
    Audio = samples,
    SampleRate = 44100,
});

var info = runtime.AnalyzeRequest(new SvcInferenceRequest {
    Audio = samples,
    SampleRate = 44100,
});

var conditioning = runtime.PrepareConditioning(new SvcInferenceRequest {
    Audio = samples,
    SampleRate = 44100,
});

var prepared = runtime.RenderPrepared(conditioning);

var direct = runtime.Render(new SvcInferenceRequest {
    Audio = samples,
    SampleRate = 44100,
});

var hostService = new SvcHostRenderService(runtime);
var hostPrepared = hostService.Prepare(new SvcHostRenderRequest {
    RequestId = "phrase-001",
    CacheKey = "track-a:phrase-001",
    Audio = samples,
    SampleRate = 44100,
});
var hostResult = hostService.Render(hostPrepared);
```

### 如何导出 ONNX ？

准备工作：
```bash
# 克隆仓库
git clone https://github.com/myueqf/DDSP-SVC.git

# 安装依赖
pip install -r requirements.txt
```

#### 导出 DDSP encoder / reflow velocity
```bash
python export_onnx.py -m <model_ckpt.pt> -o <output_dir>
```

这个脚本会输出：

- `encoder.onnx`
- `velocity.onnx`
- `svc.json`

常用附加参数：

```bash
python export_onnx.py -m <model_ckpt.pt> -o <output_dir> --skip-check
python export_onnx.py -m <model_ckpt.pt> -o <output_dir> --check-steps 20
```

说明：

- `--skip-check` 用于跳过导出后的 ONNXRuntime smoke check （如果你没装onnxruntime的话。。。）
- `--check-steps` 用于控制 smoke check 时使用的 Euler 步数

#### 导出 ContentVec ONNX

普通的 `contentvec`：

```bash
python export_contentvec_onnx.py \
  -m pretrain/contentvec/checkpoint_best_legacy_500.pt \
  -o <output_dir>/contentvec.onnx \
  --variant base
```

预导出 `tta2x` 的：

```bash
python export_contentvec_onnx.py \
  -m pretrain/contentvec/checkpoint_best_legacy_500.pt \
  -o <output_dir>/contentvec_tta2x.onnx \
  --variant tta2x
```

脚本还支持：

- `--opset 17`
- `--metadata <path>`

导出后会生成：

- `contentvec.onnx` 或 `contentvec_tta2x.onnx`
- 对应的 JSON sidecar，用于记录 `sample_rate / hop_size / opset / variant`

### 尚未实现的功能XwX

- `formant_shift_key`
- `vocal_register_shift_key`

这两个功能当前会明确报错，不会静默忽略。  
（所以是真的不支持嗷QAQ）

- `DirectML`

恐怕永远不会做XwX（需要gpu加速或许可以去康康cuda分支。。？）
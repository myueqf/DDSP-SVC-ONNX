# DDSP-SVC-ONNX

> [!WARNING]
> **温馨提示：** 此项目含有一些llm生成的代码，如有任何不适（如：头昏脑胀，血压升高，心跳加速，并且伴随着口吐白沫，浑身抽搐）请立即删除本项目并尽快睡一觉忘掉这个项目XwX

### 构建和运行  
要求：
- .NET SDK 8.0 或更高版本
- 用于 CLI 输入的 16 位 PCM WAV 文件
- 一个包含 encoder.onnx、velocity.onnx 和 svc.json 的 DDSP-SVC ONNX 导出目录
- 依赖模型：contentvec、rmvpe 以及 pc_nsf_hifigan

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
- 从 `config.yaml` / `svc.json` 读到的运行参数
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

### 尚未实现的功能XwX

- `formant_shift_key`
- `vocal_register_shift_key`

这两个功能当前会明确报错，不会静默忽略。  
（所以是真的不支持嗷QAQ）
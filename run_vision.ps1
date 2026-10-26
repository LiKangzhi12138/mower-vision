param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('env','validate','check-model','prepare','prepare-obstacle','verify-obstacle','provenance','train','evaluate','obstacle-eval','predict','export','split')]
    [string]$Task,
    [string]$Config,
    [string]$Model,
    [string]$Data,
    [string]$Split,
    [string]$Source,
    [string]$Out,
    [string]$Manifest,
    [string]$ExperimentId,
    [double]$Conf = -1,
    [double]$Iou = -1,
    [int]$ImageSize = 0,
    [switch]$CudaConv,
    [switch]$DryRun,
    [switch]$AllowTest
)
$ErrorActionPreference = 'Stop'
$expectedName = 'mower-vision'
if (-not $env:CONDA_PREFIX -or
    -not [string]::Equals((Split-Path -Leaf $env:CONDA_PREFIX.TrimEnd('\')), $expectedName, [StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "请先执行 conda activate mower-vision，再运行 .\run_vision.ps1 -Task $Task（当前 CONDA_PREFIX: $env:CONDA_PREFIX）"
    exit 2
}
$python = Join-Path $env:CONDA_PREFIX 'python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Error "找不到环境 Python: $python"
    exit 2
}
$scripts = @{
    'env'='check_env.py'; 'validate'='validate_dataset.py'; 'check-model'='check_model.py'
    'prepare'='prepare_dataset.py'; 'prepare-obstacle'='prepare_obstacle_dataset.py'
    'verify-obstacle'='prepare_obstacle_dataset.py'; 'provenance'='audit_provenance.py'
    'train'='train.py'; 'evaluate'='evaluate.py'; 'obstacle-eval'='evaluate_obstacle.py'; 'predict'='predict.py'
    'export'='export.py'; 'split'='split_dataset.py'
}
$arguments = [System.Collections.Generic.List[string]]::new()
$arguments.Add((Join-Path $PSScriptRoot ('scripts\' + $scripts[$Task])))
if ($Task -eq 'verify-obstacle') { $arguments.Add('--verify-only') }
if ($Config) { $arguments.Add('--config'); $arguments.Add($Config) }
if ($Model) { $arguments.Add('--model'); $arguments.Add($Model) }
if ($Data) { $arguments.Add('--data'); $arguments.Add($Data) }
if ($Split) { $arguments.Add('--split'); $arguments.Add($Split) }
if ($Source) { $arguments.Add('--source'); $arguments.Add($Source) }
if ($Out) { $arguments.Add($(if ($Task -eq 'split') { '--output' } else { '--out' })); $arguments.Add($Out) }
if ($Manifest) { $arguments.Add('--manifest'); $arguments.Add($Manifest) }
if ($ExperimentId) { $arguments.Add('--experiment-id'); $arguments.Add($ExperimentId) }
if ($Conf -ge 0) { $arguments.Add('--conf'); $arguments.Add($Conf.ToString([Globalization.CultureInfo]::InvariantCulture)) }
if ($Iou -ge 0) { $arguments.Add('--iou'); $arguments.Add($Iou.ToString([Globalization.CultureInfo]::InvariantCulture)) }
if ($ImageSize -gt 0) { $arguments.Add('--imgsz'); $arguments.Add([string]$ImageSize) }
if ($CudaConv) { $arguments.Add('--cuda-conv') }
if ($DryRun) { $arguments.Add('--dry-run') }
if ($AllowTest) { $arguments.Add('--allow-test') }
$before = $env:PATH
$beforeAutoInstall = $env:YOLO_AUTOINSTALL
$exitCode = 1
try {
    $entries = $before -split ';' | Where-Object {
        $_ -and -not ($_.Trim(' ', '"', '\') -match '^(?i)C:\\Program Files\\NVIDIA\\CUDNN(?:\\|$)')
    }
    $env:PATH = $entries -join ';'
    $env:YOLO_AUTOINSTALL = 'False'
    & $python @arguments
    $exitCode = $LASTEXITCODE
} catch {
    Write-Error $_
    $exitCode = 1
} finally {
    $env:PATH = $before
    $env:YOLO_AUTOINSTALL = $beforeAutoInstall
}
exit $exitCode

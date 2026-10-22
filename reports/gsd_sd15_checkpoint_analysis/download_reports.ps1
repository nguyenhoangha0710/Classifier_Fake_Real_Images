$ErrorActionPreference = 'Stop'
$dest = Join-Path $PSScriptRoot 'source'
$files = @(
 'completion.json', 'config.json', 'environment.json', 'format_counts.csv',
 'metrics.csv', 'macro_all.csv', 'macro_unseen.csv', 'paired_summary.csv',
 'predictions.csv', 'image_audit.csv',
 'seed42/split_summary.json', 'seed42/train_manifest.csv',
 'seed42/validation_manifest.csv', 'seed42/test_manifest.csv',
 'seed42/clip_gsd/history.json', 'seed42/clip_gsd/selected_checkpoint.json',
 'seed42/clip_gsd/training_status.json', 'seed42/clip_gsd/architecture.json'
)
foreach ($file in $files) {
 $target = Join-Path $dest $file
 New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
 $url = 'https://www.kaggle.com/api/v1/datasets/download/hadinho0710/checkpoint-gsd-sdv5-train/' + [Uri]::EscapeDataString('clip_gsd_tiny_sd15/' + $file)
 Invoke-WebRequest -Uri $url -OutFile $target
 Write-Output $file
}

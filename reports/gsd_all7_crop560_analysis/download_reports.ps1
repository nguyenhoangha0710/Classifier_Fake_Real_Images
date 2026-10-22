$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$dest = Join-Path $PSScriptRoot 'source'
$files = @('completion.json','config.json','environment.json','format_counts.csv','dataset_counts.csv',
 'macro_all.csv','metrics.csv','predictions.csv','token_census.csv',
 'seed42/split_summary.json','seed42/train_manifest.csv','seed42/validation_manifest.csv','seed42/test_manifest.csv',
 'seed42/train_counts.csv','seed42/validation_counts.csv','seed42/test_counts.csv',
 'seed42/clip_gsd/architecture.json','seed42/clip_gsd/history.json',
 'seed42/clip_gsd/selected_checkpoint.json','seed42/clip_gsd/training_status.json')
New-Item -ItemType Directory -Path $dest -Force | Out-Null
$metaUrl = 'https://www.kaggle.com/api/v1/datasets/view/hadinho0710/checkpoint-gsd-sdv5-train'
Invoke-WebRequest -Uri $metaUrl -OutFile (Join-Path $dest 'dataset_metadata.json')
$meta = Get-Content (Join-Path $dest 'dataset_metadata.json') -Raw | ConvertFrom-Json
if ($meta.currentVersionNumber -ne 2) { throw 'Dataset version changed; inspect before downloading.' }
foreach ($file in $files) {
 $target = Join-Path $dest $file
 New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
 $url = 'https://www.kaggle.com/api/v1/datasets/download/hadinho0710/checkpoint-gsd-sdv5-train/' + [Uri]::EscapeDataString('gsd_raw_all7_crop560_v1/' + $file) + '?datasetVersionNumber=2'
 Invoke-WebRequest -Uri $url -OutFile $target
 Write-Output $file
}

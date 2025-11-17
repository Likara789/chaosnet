# Set source and destination paths
$sourceDir = "d:\\Hell"
$destDir = "d:\\Hell\\future_repo_crap"

# Function to create directory if it doesn't exist
function Ensure-Directory {
    param([string]$path)
    if (-not (Test-Path -Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

# Copy chaosnet package
Write-Host "Copying chaosnet package..."
$chaosnetDirs = @("core", "io", "sim", "training", "utils")
foreach ($dir in $chaosnetDirs) {
    $sourcePath = Join-Path $sourceDir "chaosnet\\$dir"
    $destPath = Join-Path $destDir "chaosnet\\$dir"
    Ensure-Directory $destPath
    Copy-Item "$sourcePath\\*.py" -Destination $destPath -Force
}

# Copy root chaosnet files
Copy-Item "$sourceDir\\chaosnet\\*.py" -Destination "$destDir\\chaosnet\\" -Force

# Copy examples
Write-Host "Copying examples..."
Ensure-Directory "$destDir\\examples"
Copy-Item "$sourceDir\\examples\\*.py" -Destination "$destDir\\examples\\" -Force

# Copy visualize
Write-Host "Copying visualization files..."
Ensure-Directory "$destDir\\visualize"
Copy-Item "$sourceDir\\visualize\\*.py" -Destination "$destDir\\visualize\\" -Force

# Copy main scripts
Write-Host "Copying main scripts..."
$mainScripts = @(
    "multimodel_trainin.py",
    "train_addition.py",
    "train_language.py",
    "train_mnist_sleep.py",
    "plot_training.py",
    "generate_visualizations.py"
)
foreach ($script in $mainScripts) {
    Copy-Item "$sourceDir\\$script" -Destination "$destDir\\" -Force
}

# Create __init__.py files
Write-Host "Creating __init__.py files..."
$initDirs = @(
    "chaosnet",
    "chaosnet\\core",
    "chaosnet\\io",
    "chaosnet\\sim",
    "chaosnet\\training",
    "chaosnet\\utils",
    "examples",
    "visualize"
)
foreach ($dir in $initDirs) {
    $initPath = Join-Path $destDir "$dir\\__init__.py"
    if (-not (Test-Path $initPath)) {
        New-Item -ItemType File -Path $initPath -Force | Out-Null
    }
}

Write-Host "\nFile copy completed successfully!"
Write-Host "Run 'python update_imports.py' to update the imports in all Python files."

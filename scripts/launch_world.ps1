<#
.SYNOPSIS
    Runs the CARLA environment setup script within a specified Conda environment.

.DESCRIPTION
    This script locates the Python setup script in the current directory, activates the designated Conda environment, and executes the setup script using that environment. It ensures the correct dependencies and Python configuration are used when initializing CARLA.
#>

# Set the paths to the script location, setup script, and Conda environment
$ScriptPath = $PSScriptRoot
$EnvPath = "C:\Users\Eng05\Desktop\CARLA_0.9.15\client\env"
# $EnvPath = "C:\Users\User\Desktop\CARLA_Amur\client\env"
$PythonSetupScript = Join-Path $ScriptPath "setup_environment.py"

# Change the current working directory to the script folder
Set-Location $ScriptPath

# Execute the CARLA setup script using the specified Conda environment
conda run -p $EnvPath python $PythonSetupScript


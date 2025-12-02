<#
.SYNOPSIS
    Launches the CARLA spawn point visualization script within a specified Conda environment.

.DESCRIPTION
    This script activates the designated Conda environment and executes the visualization script.It connects to a running CARLA server to display all available spawn points on the current map and positions the spectator for a top-down overview. This ensures the correct environment and dependencies are used for the visualization process.
#>

# Set the paths to the script location, setup script, and Conda environment
$ScriptPath = $PSScriptRoot
# $EnvPath = "C:\Users\Eng05\Desktop\CARLA_0.9.15\client\env"
$EnvPath = "C:\Users\User\Desktop\CARLA_Amur\client\env"
$PythonSetupScript = Join-Path $ScriptPath "visualize_spawn_points.py"

# Change the current working directory to the script folder
Set-Location $ScriptPath

# Execute the CARLA setup script using the specified Conda environment
conda run -p $EnvPath python $PythonSetupScript
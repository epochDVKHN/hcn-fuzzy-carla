<#
.SYNOPSIS
    Launches the CARLA simulator server with configurable settings.

.DESCRIPTION
    This script sets up the CARLA server environment and starts the server executable with specified command-line arguments for window mode, resolution, graphics quality, and GPU preference. It can be easily modified for fullscreen, off-screen rendering, or different graphics APIs.
#>

# Set the paths to the CARLA server folder and executable
$ServerPath = "C:\Users\Eng05\Desktop\CARLA_0.9.15\server\WindowsNoEditor"
$ExePath = Join-Path $ServerPath "CarlaUE4.exe"

# Change the current working directory to the server folder
Set-Location $ServerPath

# Launch the CARLA server with the specified command-line arguments
Start-Process $ExePath -ArgumentList "-windowed", "-ResX=820", "-ResY=600", "-quality-level=Low", "-dx12", "-prefernvidia"

# Command-line argument options:
# - "-windowed"        : Run in windowed mode (replace with "-fullscreen" for fullscreen)
# - "-ResX" and "-ResY": Set resolution (e.g., 820x600)
# - "-quality-level"   : Set graphics quality ("Low", "High", "Epic")
# - "-dx12"            : Use DirectX 12 (replace with "-opengl" or "-vulkan" if desired)
# - "-RenderOffScreen" : Enable off-screen rendering (headless mode)
# - "-prefernvidia"    : Prefer NVIDIA GPU over integrated graphics

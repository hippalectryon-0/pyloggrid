### Comment for self:
### To change on new python version: python_v, dl_url, name in Move-Item and Remove-Item

$python_v = "3.13"  # Do not change if you want it to be automatically installed, the URL is hardcoded for Windows
$python_cmd = ".\python$( $python_v )/python.exe"
$venv = ".venv"

# Check if Chocolatey is installed
if (!(Get-Command choco -ErrorAction SilentlyContinue))
{
    Write-Host "Chocolatey is not installed. Installing Chocolatey..."

    # Prompt the user to run the script as an administrator
    if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))
    {
        Write-Host "Please provide admin credentials to install Chocolatey"
        $arguments = "& '" + $MyInvocation.MyCommand.Path + "'"
        Start-Process powershell -Verb runAs -ArgumentList $arguments
        return
    }

    # Install Chocolatey
    Set-ExecutionPolicy Bypass -Scope Process -Force
    iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
}

# Check if Mingw is installed
if (!(Get-Command mingw32-make -ErrorAction SilentlyContinue))
{
    Write-Host "Mingw is not installed. Installing Mingw via Chocolatey..."

    # Prompt the user to run the script as an administrator
    if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))
    {
        Write-Host "Please provide admin credentials to install Mingw via Chocolatey"
        $arguments = "& '" + $MyInvocation.MyCommand.Path + "'"
        Start-Process powershell -Verb runAs -ArgumentList $arguments
        return
    }

    # Install Mingw via Chocolatey
    choco install mingw -y
    choco install make -y
}

Write-Host "Mingw is installed and ready to use"

## Install python
if (-Not(Test-Path -Path "python$python_v") -and $args[0] -ne "-s")
{
    $dl_url = "https://github.com/winpython/winpython/releases/download/11.4.20250119/Winpython64-3.13.1.1dotb2.exe"

    "Downloading portable python $python_v for Windows"
    Start-BitsTransfer -Source $dl_url -Destination pythonarchive.exe
    "Downloaded portable python $python_v"
    "Unpacking"
    .\pythonarchive.exe -y | Out-Null
    Remove-Item pythonarchive.exe
    Move-Item WPy64-31311b2/python-3.13.1.amd64 "python$( $python_v )_"
    Remove-Item WPy64-31311b2 -Recurse
    Move-Item "python$( $python_v )_" "python$python_v"
    "Unpacked"
    "Testing python"
    Invoke-Expression "& .\python$( $python_v )/python.exe -V"
    "Installing pip"
    Start-BitsTransfer -Source https://bootstrap.pypa.io/get-pip.py -Destination get_pip.py
    Invoke-Expression "& $python_cmd get_pip.py --no-warn-script-location"
    Remove-Item get_pip.py
    "Python installed"
}

### Create venv

### Install python packages & enter venv
"Installing python packages"
Invoke-Expression "& $python_cmd -m pip install --upgrade pip setuptools"
Invoke-Expression "& $python_cmd -m pip install --upgrade poetry"
Invoke-Expression "& $python_cmd -m poetry config virtualenvs.in-project true"
Invoke-Expression "& $python_cmd -m virtualenv $venv"
$activate_script = Join-Path $venv "Scripts\Activate.ps1"
Invoke-Expression "& $activate_script"
Invoke-Expression "& python$python_v/Scripts/poetry install --with=docs,examples -v"
Invoke-Expression "& python -m pip uninstall pyloggrid -y"  # If you're installing from here you probably want to work with the local source and not the one in site-packages
Invoke-Expression "& pre-commit install"
Remove-Item build -Recurse
"Python packages installed"


## Compile convolver
"Compiling convolver"
cd pyloggrid/LogGrid
Invoke-Expression "& make -f Makefile.windows clean"
Invoke-Expression "& make -f Makefile.windows"
cd ../../
"Finished compiling convolver"

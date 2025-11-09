# Python Environments & Package Management

**Why this matters:** Understanding environments prevents "works on my machine" problems and dependency hell.

---

## What IS Anaconda?

**Short answer:** A Python distribution and package manager designed for data science and ML.

**Long answer:** Three related but distinct things:

### **1. Anaconda (Full Distribution)**
```
Size: ~3-5 GB
Contains:
- Python
- 250+ pre-installed packages (NumPy, Pandas, Matplotlib, Jupyter, etc.)
- Conda package manager
- Anaconda Navigator (GUI)

Use when: You want everything pre-installed
Downside: Huge download, lots you don't need
```

### **2. Miniconda (What We Installed)**
```
Size: ~50 MB
Contains:
- Python
- Conda package manager
- That's it!

Use when: You want to install only what you need
Advantage: Small, fast, clean
What we chose: Miniconda
```

### **3. Conda (The Package Manager)**
```
The tool itself (like npm for Node.js, pip for Python)
Manages: packages + environments
Works with: Python packages AND system libraries (like CUDA)
```

---

## What IS a Conda Environment?

**Scientific principle:** Isolation and reproducibility

### **The Problem Without Environments**

```
You install packages globally:
pip install tensorflow==2.10
pip install torch==2.0

Problem 1 (Conflicts):
tensorflow needs numpy==1.23
torch needs numpy==1.24
Result: One breaks!

Problem 2 (Pollution):
Project A needs old versions
Project B needs new versions
Can't have both at once!

Problem 3 (System damage):
Install wrong package
Breaks system Python
Operating system tools fail
```

### **The Solution: Virtual Environments**

```
Each project gets its own isolated environment:

Project A:
├── Python 3.10
├── tensorflow==2.10
├── numpy==1.23
└── (isolated from others)

Project B:
├── Python 3.11
├── torch==2.0
├── numpy==1.24
└── (isolated from others)

System Python: Untouched, safe
```

**Analogy:**
- Without environments: One kitchen, everyone's ingredients mixed together
- With environments: Each chef gets their own kitchen with exactly what they need

---

## Conda Environment vs venv

**Both solve the same problem (isolation), but differently:**

### **venv (Python Built-in)**

**What it is:**
```python
# Create
python -m venv myenv

# Activate (Windows)
myenv\Scripts\activate

# Install packages
pip install numpy
```

**What it does:**
- Creates isolated Python environment
- Copies Python interpreter
- Isolates packages only
- Uses pip for installation

**Advantages:**
- ✅ Built into Python (no extra install)
- ✅ Lightweight
- ✅ Fast to create
- ✅ Simple

**Limitations:**
- ❌ Python packages only (no system libraries)
- ❌ Can't install different Python versions easily
- ❌ No CUDA, no compiled libraries
- ❌ Doesn't track system dependencies

### **Conda Environments**

**What it is:**
```bash
# Create
conda create -n myenv python=3.10

# Activate
conda activate myenv

# Install packages
conda install numpy
# OR
pip install numpy  # conda environments support pip too!
```

**What it does:**
- Creates isolated environment
- Installs Python version you specify
- Manages Python packages AND system libraries
- Can install CUDA, compilers, non-Python tools

**Advantages:**
- ✅ Can install any Python version (3.7, 3.10, 3.11, etc.)
- ✅ Handles system libraries (CUDA, MKL, compilers)
- ✅ Cross-platform consistency
- ✅ Can use pip AND conda
- ✅ Solves complex dependencies better

**Limitations:**
- ❌ Requires Anaconda/Miniconda installed
- ❌ Slower to create (downloads more)
- ❌ Takes more disk space

### **Why We Use Conda for ML**

**ML projects need system libraries:**
```
PyTorch needs:
- Python 3.10
- CUDA 12.1 (GPU drivers)
- cuDNN (CUDA neural network library)
- MKL (Intel math kernel)
- C++ compilers

venv: Can only install PyTorch Python wrapper, breaks without system libs
conda: Installs ALL of the above, guaranteed to work together
```

**The real power:**
```bash
# With conda, this just works:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# It installs:
- PyTorch Python package
- CUDA 12.1 libraries
- cuDNN
- All dependencies
- All guaranteed compatible!

# With venv + pip:
pip install torch
# Result: Maybe works, maybe doesn't, depends on what's already installed
# Need to manually install CUDA, hope versions match
```

---

## What Did We Just Install?

### **Step 1: Miniconda Base**

```
Downloaded: 50 MB
Installed to: C:\Users\daniel.hicks\AppData\Local\miniconda3\

What's inside:
├── python.exe (base Python 3.x)
├── conda.exe (package manager)
├── Scripts\
│   ├── activate.bat (environment activation)
│   ├── pip.exe (Python package installer)
│   └── conda.exe
├── Library\
│   ├── bin\ (system binaries)
│   └── include\ (C headers for compiling)
└── pkgs\ (package cache)
```

**This is the "base" environment** - we DON'T install packages here!

### **Step 2: Accepted Terms of Service**

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
```

**What this does:**
- Accepts Anaconda's usage terms
- Allows downloading from official repositories
- Three channels:
  - main: Primary packages (numpy, pandas, python)
  - r: R language packages
  - msys2: Unix tools for Windows

**Why needed:**
- Legal requirement to use Anaconda repositories
- One-time per installation

### **Step 3: Created "persona" Environment**

```bash
conda create -n persona python=3.10 -y
```

**What happened:**

**Downloaded (18.1 MB total):**
```
python-3.10.19        15.3 MB   The Python interpreter itself
setuptools-80.9.0      1.4 MB   Package installation tools
pip-25.2               1.0 MB   Python package installer
sqlite-3.51.0          0.9 MB   Database (Python uses internally)
openssl-3.0.18         0.5 MB   Cryptography library
libffi-3.4.4           0.2 MB   Foreign function interface (calling C from Python)
bzip2, zlib, xz        <1 MB    Compression libraries
ca-certificates        <1 MB    SSL certificates
```

**Created isolated environment at:**
```
C:\Users\daniel.hicks\AppData\Local\miniconda3\envs\persona\

persona/
├── python.exe (Python 3.10.19, isolated from base)
├── Scripts\
│   ├── pip.exe (for installing packages in THIS environment)
│   └── activate.bat
├── Lib\
│   ├── site-packages\ (where packages go)
│   └── (standard library)
└── Include\ (C headers)
```

**Why these packages:**

**Python 3.10.19:**
- The interpreter itself
- 3.10 chosen for compatibility (PyTorch, Unsloth tested on this)
- Newer (3.14 on your system) might have issues
- Older (3.7, 3.8) missing features we need

**setuptools:**
- Builds and installs Python packages
- Needed to compile packages with C extensions
- PyTorch has C++/CUDA code that needs compilation

**pip:**
- Python Package Installer
- Downloads from PyPI (Python Package Index)
- Conda can use pip for packages not in conda repositories

**OpenSSL:**
- Cryptography library
- HTTPS downloads (secure package installation)
- SSL/TLS for network communications

**sqlite:**
- Lightweight database
- Python standard library uses it
- Package managers cache metadata in sqlite

**libffi:**
- Foreign Function Interface
- Lets Python call C/C++ functions
- Critical for ML libraries (mostly written in C++/CUDA)

**Compression libraries (bzip2, zlib, xz):**
- Decompress downloaded packages
- Packages are compressed to save bandwidth
- Python modules might need them

**ca-certificates:**
- Certificate Authority certificates
- Verify HTTPS connections are legitimate
- Prevents man-in-the-middle attacks during package downloads

### **Step 4: Initialized Conda for PowerShell**

```bash
conda init powershell
```

**What this did:**

**Modified your PowerShell profile:**
```
File: C:\Users\daniel.hicks\OneDrive - Helix Systems Inc\Documents\PowerShell\profile.ps1

Added:
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
$Env:CONDA_EXE = "C:\Users\daniel.hicks\AppData\Local\miniconda3\Scripts\conda.exe"
$Env:_CE_M = ""
$Env:_CE_CONDA = ""
$Env:_CONDA_ROOT = "C:\Users\daniel.hicks\AppData\Local\miniconda3"
$Env:_CONDA_EXE = "C:\Users\daniel.hicks\AppData\Local\miniconda3\Scripts\conda.exe"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs
Remove-Variable CondaModuleArgs
# <<< conda initialize <<<
```

**What this does:**
- Runs every time you open PowerShell
- Sets up conda commands (`conda activate`, `conda install`, etc.)
- Shows current environment in prompt: `(base)` or `(persona)`
- Without this: `conda activate` doesn't work

**Why it needs restart:**
- PowerShell profile runs on startup
- Current terminal doesn't have it loaded yet
- New terminal will run the profile

---

## Understanding the Environment Structure

### **Directory Layout**

```
miniconda3/
├── (base environment - don't touch)
│
├── envs/                    ← All your environments here
│   ├── persona/             ← Our project environment
│   │   ├── python.exe
│   │   ├── Scripts/
│   │   ├── Lib/
│   │   └── site-packages/
│   │
│   ├── other-project/       ← Could create another
│   └── another-env/         ← And another
│
└── pkgs/                    ← Package cache (shared)
    ├── python-3.10.19/
    ├── numpy-1.24.3/
    └── torch-2.0.1/
```

**Key insight: Environments are isolated, but packages are cached!**

```
First time: Download numpy-1.24.3 (50 MB)
Create new env: Copy from cache (instant)
Result: Fast environment creation, no redundant downloads
```

### **How Activation Works**

**Before activation:**
```powershell
PS> python --version
Python 3.14.0  # System Python

PS> where python
C:\Users\daniel.hicks\AppData\Local\Programs\Python\Python314\python.exe
```

**After activation:**
```powershell
PS> conda activate persona
(persona) PS> python --version
Python 3.10.19  # Environment Python!

(persona) PS> where python
C:\Users\daniel.hicks\AppData\Local\miniconda3\envs\persona\python.exe
C:\Users\daniel.hicks\AppData\Local\Programs\Python\Python314\python.exe

# First one in PATH wins!
```

**What `conda activate` does:**
1. Prepends environment's `Scripts\` to PATH
2. Sets environment variables (`CONDA_DEFAULT_ENV=persona`)
3. Changes prompt to show active environment
4. All commands now use environment's tools

**Deactivation:**
```powershell
(persona) PS> conda deactivate
PS>  # Back to base/system
```

---

## Practical Examples

### **Creating Multiple Environments**

```bash
# Different Python versions
conda create -n old-project python=3.7
conda create -n new-project python=3.11

# Different package versions
conda create -n tensorflow-env tensorflow=2.10
conda create -n pytorch-env pytorch=2.0

# Clone an environment
conda create -n persona-backup --clone persona
```

### **Managing Environments**

```bash
# List all environments
conda env list

# Delete an environment
conda env remove -n old-project

# Export environment (for reproducibility)
conda env export > environment.yml

# Recreate environment from file
conda env create -f environment.yml
```

### **Installing Packages**

```bash
# With conda (preferred for ML libraries)
conda install numpy pandas matplotlib

# With pip (for packages not in conda)
pip install transformers unsloth

# Specific versions
conda install pytorch=2.0.1
pip install numpy==1.24.3

# From specific channel
conda install pytorch -c pytorch
```

---

## Why This Matters for This Project

### **What We're About to Do**

```bash
conda activate persona  # Use our isolated environment

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install other dependencies
pip install transformers datasets accelerate
```

**If we didn't use conda:**
- Might break system Python
- CUDA version conflicts
- Can't easily test different Python versions
- Harder to share setup with others
- Difficult to recreate environment if something breaks

**With conda:**
- Isolated (can't break anything outside)
- CUDA installed correctly
- Can have multiple projects with different requirements
- Easy to share: `conda env export > environment.yml`
- Easy to recreate: `conda env create -f environment.yml`

---

## Common Commands Reference

```bash
# Create environment
conda create -n myenv python=3.10

# Activate/deactivate
conda activate myenv
conda deactivate

# Install packages
conda install package_name
pip install package_name

# List installed packages
conda list
pip list

# Update packages
conda update package_name
pip install --upgrade package_name

# Remove packages
conda remove package_name
pip uninstall package_name

# Environment management
conda env list           # List all environments
conda env remove -n name # Delete environment
conda env export         # Export environment.yml

# Clean up (free disk space)
conda clean --all
```

---

## Summary

**What is Anaconda/Miniconda?**
- Distribution of Python + package manager
- Miniconda = minimal version (what we installed)
- Conda = the package manager tool

**What did we install?**
- Miniconda base (50 MB)
- "persona" environment with Python 3.10 (18 MB)
- Essential packages: pip, setuptools, OpenSSL, compression libs

**What is a conda environment?**
- Isolated Python installation
- Own packages, own Python version
- Similar to venv but more powerful (handles system libraries)

**Why conda for ML?**
- Installs CUDA, compiled libraries
- Handles complex dependencies
- Cross-platform consistency
- Guaranteed compatibility

**Next step:** Activate the environment and install PyTorch + Unsloth!

---

*Next: [Module 3: Neural Networks](03_neural_networks.md) (coming soon)*

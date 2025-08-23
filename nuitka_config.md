# Nuitka Build Configuration for NeuralMail

This document describes the Nuitka configuration used to build the NeuralMail standalone executable.

## Build Command Used:

```bash
python -m nuitka \
    --standalone \
    --windows-disable-console \
    --windows-icon-from-ico=icon.ico \
    --enable-plugin=pyqt5 \
    --enable-plugin=numpy \
    --include-data-files=config.py=config.py \
    --include-data-files=icon.ico=icon.ico \
    --output-dir=dist \
    --assume-yes-for-downloads \
    --show-progress \
    --mingw64 \
    --lto=no \
    neuralmail.py
```

## Key Options Explained:

- `--standalone`: Creates a folder with executable and all dependencies
- `--windows-disable-console`: GUI application (no console window)
- `--windows-icon-from-ico`: Uses the application icon
- `--enable-plugin=pyqt5`: Enables PyQt5 support
- `--enable-plugin=numpy`: Enables NumPy optimization
- `--include-data-files`: Includes configuration and icon files
- `--mingw64`: Uses MinGW64 compiler on Windows
- `--lto=no`: Disables Link Time Optimization for faster builds

## Security Features 🔒

The build process includes several security measures to prevent sensitive data from being included in the executable:

### Build Script Security:
The `build_executable.py` script automatically:
- Scans for sensitive files before building
- Lists all detected sensitive files with sizes
- Confirms they will be excluded from the build
- Requires user confirmation to continue
- Only copies safe files to distribution

### Excluded Files (via build script logic):
- `*.db` - Email database files (can be 100MB+ and contain private emails)
- `*.log` - Log files that may contain sensitive information
- `config.json*` - Actual configuration files with API keys and credentials
- `background_info.txt` - User profile information
- `research_logs/` - Research logs that may contain private data

### Safe Files Included:
- `config.py` - Contains only default configuration template (no secrets)
- `icon.ico` - Application icon
- `README.md` & `LICENSE` - Documentation files

**Note**: Security is enforced by the build script, not Nuitka command-line options. The script ensures no sensitive files are copied to the distribution folder.

## Requirements:

### Windows:
- Python 3.8+
- A C++ compiler (one of):
  - Microsoft Visual Studio (Build Tools or Community Edition)
  - MinGW64 (recommended, automatically downloaded by Nuitka)

### All Platforms:
- Nuitka (`pip install nuitka`)
- ordered-set (`pip install ordered-set`)

## Advantages of Nuitka vs PyInstaller:

1. **Performance**: Compiled to machine code, runs faster
2. **Memory Usage**: Lower memory footprint
3. **Startup Time**: Faster application startup
4. **Size**: Often produces smaller executables
5. **Compatibility**: Better Python compatibility and fewer runtime issues

## Build Output:

- `dist/neuralmail.dist/` - Folder containing the executable and all dependencies
- `dist/neuralmail.dist/neuralmail.exe` - The main executable file
- `neuralmail.build/` - Temporary build files (can be deleted)
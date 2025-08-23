#!/usr/bin/env python
"""
Build script for creating a standalone executable of neuralmail.py using Nuitka
"""
import os
import sys
import platform
import subprocess
import shutil
import importlib
import pkg_resources


def check_dependency(package_name):
    """Check if a dependency is installed and return its version."""
    try:
        package = importlib.import_module(package_name)
        version = pkg_resources.get_distribution(package_name).version
        return True, version
    except (ImportError, pkg_resources.DistributionNotFound):
        return False, None


def check_sensitive_files():
    """Check for sensitive files that should not be included in the build."""
    sensitive_patterns = [
        "config.json",
        "config.json.backup", 
        "*.db",
        "*.log",
        "background_info.txt",
        "research_logs/*"
    ]
    
    found_sensitive = []
    
    # Check for exact file matches
    for pattern in ["config.json", "config.json.backup", "background_info.txt"]:
        if os.path.exists(pattern):
            found_sensitive.append(pattern)
    
    # Check for database files
    for file in os.listdir("."):
        if file.endswith(".db") or file.endswith(".log"):
            found_sensitive.append(file)
    
    # Check research_logs directory
    if os.path.exists("research_logs") and os.listdir("research_logs"):
        found_sensitive.append("research_logs/")
    
    if found_sensitive:
        print("\n⚠️  WARNING: SENSITIVE FILES DETECTED ⚠️")
        print("The following files contain sensitive data and will NOT be included in the build:")
        for file in found_sensitive:
            size = ""
            if os.path.isfile(file):
                size = f" ({os.path.getsize(file):,} bytes)"
            print(f"  ❌ {file}{size}")
        
        print("\n✅ These files are properly excluded from the build for security.")
        print("   Users will need to configure their own settings after installation.")
        
        response = input("\nContinue with build? (y/N): ").strip().lower()
        if response != 'y':
            print("Build cancelled for security review.")
            return False
    
    return True


def main():
    # Check for sensitive files before building
    print("🔒 Checking for sensitive files...")
    if not check_sensitive_files():
        return
    
    # Check key dependencies
    dependencies = [
        "PyQt5",
        "openai",
        "numpy",
        "tiktoken",
        "html2text",
        "PyPDF2",
        "python-docx",
        "matplotlib",
        "scipy",
    ]
    missing_deps = []

    print("Checking dependencies...")
    for dep in dependencies:
        # Handle package name variations
        check_name = dep
        if dep == "python-docx":
            check_name = "docx"

        installed, version = check_dependency(check_name)
        if installed:
            print(f"✓ {dep} {version}")
        else:
            missing_deps.append(dep)
            print(f"✗ {dep} not found")

    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies with:")
        print(f"pip install {' '.join(missing_deps)}")
        response = input("\nWould you like to install them now? (y/N): ").strip().lower()
        if response == 'y':
            print("Installing missing dependencies...")
            for dep in missing_deps:
                print(f"Installing {dep}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                except subprocess.CalledProcessError:
                    print(
                        f"Failed to install {dep}. Please install it manually with 'pip install {dep}'"
                    )
                    return
        else:
            print("Please install missing dependencies and try again.")
            return

    # Check if Nuitka is installed
    print("\n🔧 Checking Nuitka installation...")
    nuitka_installed, nuitka_version = check_dependency("nuitka")
    if nuitka_installed:
        print(f"✓ Nuitka {nuitka_version}")
    else:
        print("❌ Nuitka not found")
        response = input("Would you like to install Nuitka now? (y/N): ").strip().lower()
        if response == 'y':
            print("Installing Nuitka...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "nuitka", "ordered-set"]
                )
                print("✅ Nuitka installation complete!")
            except subprocess.CalledProcessError:
                print("❌ Failed to install Nuitka. Please install it manually with: pip install nuitka ordered-set")
                return
        else:
            print("Please install Nuitka manually with: pip install nuitka ordered-set")
            return

    # Create build directory if it doesn't exist
    if not os.path.exists("dist"):
        os.makedirs("dist")

    # Clean previous builds
    print("\nCleaning previous builds...")
    if os.path.exists("neuralmail.build"):
        shutil.rmtree("neuralmail.build")
    if os.path.exists("neuralmail.dist"):
        shutil.rmtree("neuralmail.dist")
    if os.path.exists("neuralmail.exe"):
        os.remove("neuralmail.exe")
    if os.path.exists("dist"):
        shutil.rmtree("dist")

    # Create dist directory
    os.makedirs("dist", exist_ok=True)

    # Build the executable with Nuitka
    print("\nBuilding the executable with Nuitka...")

    build_command = [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",  # Create standalone executable
        "--windows-disable-console",  # GUI app (no console window)
        f"--windows-icon-from-ico=icon.ico",  # Use icon file
        "--enable-plugin=pyqt5",  # Enable PyQt5 plugin
        "--enable-plugin=numpy",  # Enable NumPy plugin
        "--include-data-files=config.py=config.py",  # Include config template (safe - no secrets)
        "--include-data-files=icon.ico=icon.ico",  # Include icon
        "--output-dir=dist",  # Output to dist folder
        "--assume-yes-for-downloads",  # Auto-download dependencies
        "--show-progress",  # Show build progress
        "neuralmail.py",  # Main script
    ]

    # Add platform-specific options
    if platform.system() == "Windows":
        build_command.extend(
            [
                "--mingw64",  # Use MinGW64 compiler on Windows
                "--lto=no",  # Disable LTO for faster builds
            ]
        )

    try:
        subprocess.check_call(build_command)
    except subprocess.CalledProcessError as e:
        print(f"\nError building executable: {e}")
        print("\nTrying with verbose output...")
        verbose_command = build_command + ["--verbose"]
        try:
            subprocess.check_call(verbose_command)
        except subprocess.CalledProcessError:
            print("\nBuild failed. Please check the output above for errors.")
            print(
                "You may need to install a C++ compiler (Microsoft Visual Studio or MinGW64)."
            )
            return

    # Copy additional files to dist folder (SAFE FILES ONLY)
    files_to_copy = ["README.md", "LICENSE"]
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, f"dist/{file}")

    # Create an instructions file alongside the executable
    instructions_content = """NEURALMAIL INSTRUCTIONS
======================

This is a standalone executable of the NeuralMail application.

To use:
1. Double-click the 'neuralmail.exe' (or 'neuralmail' on Linux/Mac) executable to start the application
2. Configure your email settings on first run
3. Set up your OpenAI API key in the configuration
4. The application will sync your emails and provide AI-powered email processing

REQUIREMENTS:
- Internet access for email synchronization and OpenAI API calls
- Valid email credentials (IMAP access)
- OpenAI API key for AI features

FIRST RUN:
- The application will create necessary configuration files
- You'll need to configure your email settings and API keys
- A database will be created to store email data locally

For any issues, please refer to the README.md file or contact the developers.

NOTE: This is a standalone executable compiled with Nuitka that includes all 
necessary Python libraries and does not require a separate Python installation. 
Nuitka-compiled executables start faster and use less memory than other bundlers.
The executable is in a folder with all its dependencies for optimal performance.
"""

    with open("dist/INSTRUCTIONS.txt", "w") as f:
        f.write(instructions_content)

    print("\nBuild complete! Nuitka executable created in the 'dist' folder.")
    print("You can distribute the entire 'dist/neuralmail.dist' folder to others.")
    print(
        "Optional: Also distribute INSTRUCTIONS.txt, README.md, and LICENSE for user reference."
    )
    print("\nTo test the executable:")
    if platform.system() == "Windows":
        print("Double-click 'dist/neuralmail.dist/neuralmail.exe'")
    else:
        print("Run 'dist/neuralmail.dist/neuralmail' in terminal")
    print(
        "\nNOTE: Folder-based distribution starts faster and uses less memory than single-file!"
    )


if __name__ == "__main__":
    main()

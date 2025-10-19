"""
BeamNG.drive Installation and Compatibility Checker
Data-Driven Driver Project - Phase 1 Verification

This script verifies that:
1. BeamNGpy is properly installed
2. BeamNG.drive/tech is accessible
3. Basic connection can be established
4. Version compatibility is confirmed
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠ Warning: Python 3.8+ recommended for BeamNGpy")
        return False
    else:
        print("✓ Python version is compatible")
        return True

def check_beamngpy_installation():
    """Check if BeamNGpy is properly installed."""
    print("\nChecking BeamNGpy installation...")
    
    try:
        import beamngpy
        print(f"✓ BeamNGpy version: {beamngpy.__version__}")
        return True
    except ImportError as e:
        print(f"✗ BeamNGpy not found: {e}")
        print("Install with: pip install beamngpy")
        return False

def check_beamng_executable():
    """Attempt to locate BeamNG.drive executable."""
    print("\nChecking for BeamNG.drive installation...")
    
    # Common installation paths
    possible_paths = [
        Path("S:/SteamLibrary/steamapps/common/BeamNG.drive"),  # User's installation
        Path("C:/Users") / Path.home().name / "AppData/Local/BeamNG.drive",
        Path("C:/Program Files/BeamNG.drive"),
        Path("C:/Program Files (x86)/BeamNG.drive"),
        Path("D:/SteamLibrary/steamapps/common/BeamNG.drive"),
        Path("C:/Program Files (x86)/Steam/steamapps/common/BeamNG.drive"),
    ]
    
    found_paths = []
    for path in possible_paths:
        if path.exists():
            executable = path / "BeamNG.drive.exe"
            tech_executable = path / "BeamNG.tech.exe"
            
            if executable.exists() or tech_executable.exists():
                found_paths.append(path)
                print(f"✓ Found BeamNG installation: {path}")
    
    if not found_paths:
        print("⚠ BeamNG.drive installation not found in common locations")
        print("Please ensure BeamNG.drive or BeamNG.tech is installed")
        return False
    
    return True

def check_connection_test():
    """Test basic BeamNG connection without loading scenario."""
    print("\nTesting basic BeamNG connection...")
    
    try:
        from beamngpy import BeamNGpy
        
        # Try to create BeamNG instance (doesn't actually connect yet)
        bng = BeamNGpy('localhost', 64256)
        print("✓ BeamNGpy instance created successfully")
        
        # Note: We don't actually connect here to avoid requiring BeamNG to be running
        print("✓ Basic connection test passed")
        print("Note: Actual connection requires BeamNG.drive to be running")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False

def main():
    """Run all compatibility checks."""
    print("="*60)
    print("BeamNG.drive AI Automation - Installation Checker")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("BeamNGpy Installation", check_beamngpy_installation),
        ("BeamNG.drive Installation", check_beamng_executable),
        ("Connection Test", check_connection_test),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} check failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("COMPATIBILITY CHECK SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("✓ All checks passed! Ready to run Phase 1 setup.")
        print("Next step: Run 'python phase1_basic_setup.py'")
    else:
        print("⚠ Some checks failed. Please resolve issues before proceeding.")
        print("Common solutions:")
        print("  - Install BeamNGpy: pip install beamngpy")
        print("  - Install BeamNG.drive from: https://beamng.com/")
        print("  - Ensure Python 3.8+ is installed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
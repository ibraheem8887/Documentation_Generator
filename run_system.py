#!/usr/bin/env python3
"""
Startup script for the Python Documentation Generation System

This script provides easy access to all system components:
- Run integration tests
- Start Streamlit web interface
- Command line documentation generation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests():
    """Run integration tests"""
    print("🧪 Running Integration Tests...")
    print("=" * 50)
    
    test_script = Path("integration") / "test_integration.py"
    
    if not test_script.exists():
        print(f"❌ Test script not found: {test_script}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(test_script)], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def start_web_ui():
    """Start Streamlit web interface"""
    print("🌐 Starting Web Interface...")
    print("=" * 50)
    
    ui_script = Path("ui") / "streamlit_app.py"
    
    if not ui_script.exists():
        print(f"❌ UI script not found: {ui_script}")
        return False
    
    try:
        # Change to UI directory and run streamlit
        os.chdir("ui")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        return True
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        return False


def run_cli_generator(args):
    """Run command line documentation generator"""
    print("💻 Running CLI Documentation Generator...")
    print("=" * 50)
    
    cli_script = Path("integration") / "doc_generator.py"
    
    if not cli_script.exists():
        print(f"❌ CLI script not found: {cli_script}")
        return False
    
    try:
        # Build command
        cmd = [sys.executable, str(cli_script)]
        
        if args.input:
            cmd.append(args.input)
        
        if args.output:
            cmd.extend(["-o", args.output])
        
        if args.string:
            cmd.append("-s")
        
        if args.model_dir:
            cmd.extend(["--model-dir", args.model_dir])
        
        # Run command
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running CLI generator: {e}")
        return False


def check_system():
    """Check system requirements and model files"""
    print("🔍 Checking System Requirements...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("⚠️ Warning: Python 3.7+ recommended")
    else:
        print("✅ Python version OK")
    
    # Check required packages
    required_packages = [
        'torch', 'numpy', 'pandas', 'streamlit', 'ast'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
    
    # Check model files
    print("\n📁 Checking Model Files...")
    model_files = [
        "bilstm/bilstm_model.pt",
        "bilstm/bpe_tokenizer.pkl",
        "word2vec/word2vec_model.pt",
        "bpe/bpe_tokenizer.py"
    ]
    
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✅ {model_file}")
        else:
            print(f"❌ {model_file} (missing)")
    
    # Check directories
    print("\n📂 Checking Directories...")
    directories = ["integration", "ui", "bpe", "word2vec", "bilstm"]
    
    for directory in directories:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ (missing)")
    
    print("\n" + "=" * 50)
    print("System check completed!")


def show_help():
    """Show help information"""
    help_text = """
🚀 Python Documentation Generation System

Available Commands:
  test        Run integration tests
  web         Start Streamlit web interface  
  cli         Run command line generator
  check       Check system requirements
  help        Show this help message

Examples:
  python run_system.py test
  python run_system.py web
  python run_system.py cli -s "def add(a,b): return a+b" -o docs.md
  python run_system.py check

For CLI options:
  python run_system.py cli --help
"""
    print(help_text)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Python Documentation Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    subparsers.add_parser('test', help='Run integration tests')
    
    # Web command
    subparsers.add_parser('web', help='Start Streamlit web interface')
    
    # CLI command
    cli_parser = subparsers.add_parser('cli', help='Run command line generator')
    cli_parser.add_argument('input', nargs='?', help='Input Python file or code string')
    cli_parser.add_argument('-o', '--output', help='Output file path')
    cli_parser.add_argument('-s', '--string', action='store_true', 
                           help='Treat input as code string instead of file path')
    cli_parser.add_argument('--model-dir', help='Directory containing trained models')
    
    # Check command
    subparsers.add_parser('check', help='Check system requirements')
    
    # Help command
    subparsers.add_parser('help', help='Show help message')
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'test':
        success = run_tests()
        sys.exit(0 if success else 1)
    
    elif args.command == 'web':
        success = start_web_ui()
        sys.exit(0 if success else 1)
    
    elif args.command == 'cli':
        if not args.input:
            print("❌ Error: Input required for CLI mode")
            print("Use: python run_system.py cli --help")
            sys.exit(1)
        
        success = run_cli_generator(args)
        sys.exit(0 if success else 1)
    
    elif args.command == 'check':
        check_system()
        sys.exit(0)
    
    elif args.command == 'help':
        show_help()
        sys.exit(0)
    
    else:
        # No command provided, show help
        show_help()
        sys.exit(0)


if __name__ == "__main__":
    main()

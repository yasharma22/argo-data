#!/usr/bin/env python3
"""
ARGO Data Platform - Main Application Entry Point

This is the main entry point for the ARGO Data Platform.
It provides a simple interface to launch the Streamlit dashboard.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the ARGO Data Platform dashboard"""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    dashboard_path = project_root / "src" / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print("❌ Dashboard application not found!")
        print(f"Expected location: {dashboard_path}")
        return 1
    
    print("🌊 Launching ARGO Data Platform Dashboard...")
    print(f"📍 Project root: {project_root}")
    print(f"🚀 Starting Streamlit server...")
    
    try:
        # Launch Streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], cwd=project_root)
        
    except KeyboardInterrupt:
        print("\n👋 ARGO Data Platform stopped.")
        return 0
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

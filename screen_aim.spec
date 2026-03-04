# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Screen Aim Assistant.

Build:
    pyinstaller screen_aim.spec

Output:
    dist/ScreenAimAssistant/
"""

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'dearpygui',
        'numpy',
        'PIL',
        'mss',
        'pynput',
        'pynput.keyboard',
        'pynput.mouse',
        'ultralytics',
        'torch',
        'torchvision',
        'onnx',
        'onnxruntime',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ScreenAimAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ScreenAimAssistant',
)

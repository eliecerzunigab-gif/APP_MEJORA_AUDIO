# PyInstaller spec para empaquetar Audio Enchancer como .exe unico
# Uso: pyinstaller build.spec
# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('templates/index.html', 'templates'),
    ],
    hiddenimports=[
        'numpy', 'scipy', 'scipy.fft', 'scipy.signal',
        'soundfile', 'mutagen', 'mutagen.mp3', 'mutagen.id3',
        'flask', 'jinja2', 'markupsafe', 'click', 'itsdangerous',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'pandas', 'PIL', 'cv2'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Audio_Enhancer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
# PyInstaller spec for the Windows build. Run from the repository root:
#   pyinstaller packaging/UVR.spec --noconfirm
# Produces dist/UVR/UVR.exe with all files next to it (UVR.py expects its data
# folders in sys._MEIPASS, which is the exe's folder with contents_directory='.').
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = os.path.dirname(SPECPATH)


def tree(src, dest=None, skip_ext=('.py', '.pyc')):
    """(file, folder) pairs for every data file under src, without Python sources or caches."""
    base = os.path.join(ROOT, src)
    pairs = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for name in filenames:
            if not name.endswith(skip_ext):
                pairs.append((os.path.join(dirpath, name), os.path.join(dest or src, os.path.relpath(dirpath, base))))
    return pairs


datas = tree('gui_data') + tree('models') + tree(os.path.join('lib_v5', 'vr_network', 'modelparams'))
datas += [(os.path.join(ROOT, 'lib_v5', 'mixer.ckpt'), 'lib_v5')]
for package in ('librosa', 'resampy', 'samplerate', 'lightning_fabric', 'pytorch_lightning'):
    datas += collect_data_files(package)
for dist in ('torch', 'tqdm', 'numpy', 'packaging', 'pytorch_lightning', 'lightning_utilities',
             'torchmetrics', 'onnxruntime-gpu', 'onnxruntime', 'librosa', 'scipy'):
    try:
        datas += copy_metadata(dist)
    except Exception:
        print(f'UVR.spec: {dist} is not installed, skipping its metadata')

# FFmpeg (MP3/FLAC output) and Rubber Band (pitch/time tools) go next to UVR.exe
bin_dir = os.path.join(SPECPATH, 'bin')
binaries = [(os.path.join(bin_dir, name), '.') for name in os.listdir(bin_dir)] if os.path.isdir(bin_dir) else []
binaries += collect_dynamic_libs('samplerate')


def local_modules(package):
    """Every module of a package in this repo. collect_submodules can't import them, and some
    (e.g. demucs.htdemucs) are only referenced from pickled model files."""
    base = os.path.join(ROOT, package)
    modules = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        prefix = os.path.relpath(dirpath, ROOT).replace(os.sep, '.')
        modules += [f'{prefix}.{name[:-3]}' for name in filenames if name.endswith('.py') and name not in ('__init__.py', '__main__.py')]
    return modules


hiddenimports = (local_modules('demucs') + local_modules('lib_v5') + local_modules('gui_data')
                 + collect_submodules('onnx2pytorch') + ['tools.headless_separate', 'pkg_resources']
                 # scikit-learn (used by librosa) imports its vendored array API modules dynamically
                 + collect_submodules('sklearn.externals'))

a = Analysis(
    [os.path.join(ROOT, 'UVR.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=['torchaudio', 'matplotlib', 'IPython', 'tensorboard', 'transformers'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='UVR',
    icon=os.path.join(ROOT, 'gui_data', 'img', 'GUI-Icon.ico'),
    console=False,
    upx=False,
    contents_directory='.',
)
coll = COLLECT(exe, a.binaries, a.datas, name='UVR', upx=False)

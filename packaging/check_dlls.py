"""Checks that every DLL the built app imports ships with it or with Windows itself.

The build machine has Visual Studio and every Visual C++ runtime installed, so an app
can start there and still crash on a normal PC that lacks one of them. Exits 1 and
lists the culprits if a DLL would only be found on the build machine.

    python packaging/check_dlls.py dist/UVR
"""
import os
import re
import sys

import pefile

# Installed by redistributables or toolkits, not by Windows: must be bundled
REDISTRIBUTABLE = re.compile(r'^(msvcp\d+.*|vcruntime\d+.*|concrt\d+|vcomp\d+|libomp\d+.*|vccorlib\d+|mfc\d+.*|msvcr\d+.*|'
                             r'libiomp5md|cudart.*|cublas.*|cudnn.*|cufft.*|curand.*|cusparse.*|cusolver.*|nvrtc.*|nvjitlink.*|tbb.*)\.dll$')
# Provided by the NVIDIA driver on machines that have one; never bundled
DRIVER = {'nvcuda.dll', 'nvml.dll', 'nvapi64.dll', 'nvcuvid.dll', 'nvencodeapi64.dll'}
SYSTEM32 = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')


def imported_dlls(path):
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],
                                           pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT']])
    names = set()
    for attr in ('DIRECTORY_ENTRY_IMPORT', 'DIRECTORY_ENTRY_DELAY_IMPORT'):
        for entry in getattr(pe, attr, []):
            names.add(entry.dll.decode('ascii', 'replace').lower())
    pe.close()
    return names


def main():
    dist = sys.argv[1]
    bundled = set()
    binaries = []
    for root, _, files in os.walk(dist):
        for name in files:
            bundled.add(name.lower())
            if name.lower().endswith(('.dll', '.pyd', '.exe')):
                binaries.append(os.path.join(root, name))

    problems = {}
    for path in binaries:
        try:
            dlls = imported_dlls(path)
        except pefile.PEFormatError:
            continue
        for dll in dlls:
            if dll in bundled or dll in DRIVER or dll.startswith(('api-ms-win-', 'ext-ms-')):
                continue
            in_system32 = os.path.isfile(os.path.join(SYSTEM32, dll))
            if REDISTRIBUTABLE.match(dll) or not in_system32:
                where = 'build machine System32 only' if in_system32 else 'not found anywhere'
                problems.setdefault(f'{dll} ({where})', []).append(os.path.relpath(path, dist))

    print(f'Checked {len(binaries)} binaries in {dist}')
    for dll, users in sorted(problems.items()):
        print(f'MISSING {dll}: needed by {", ".join(sorted(users)[:5])}{" ..." if len(users) > 5 else ""}')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()

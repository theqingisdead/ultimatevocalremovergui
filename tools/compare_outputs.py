"""Compare stems from two headless_separate.py runs (e.g. upstream vs this fork).

For each WAV with the same name in both folders, prints the largest sample
difference and the signal-to-difference ratio in dB. "identical" means
bit-exact; above ~100 dB is float rounding noise; below ~60 dB is audible
territory and worth listening to.

    python tools/compare_outputs.py out/upstream out/fork
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf


def compare(ref_path, test_path):
    ref, ref_sr = sf.read(ref_path, dtype='float64', always_2d=True)
    test, test_sr = sf.read(test_path, dtype='float64', always_2d=True)
    if ref_sr != test_sr or ref.shape != test.shape:
        return f'shape/rate mismatch: {ref.shape}@{ref_sr} vs {test.shape}@{test_sr}', None
    diff = ref - test
    max_diff = float(np.abs(diff).max())
    if max_diff == 0:
        return 'identical', float('inf')
    snr = 10 * np.log10(np.sum(ref ** 2) / np.sum(diff ** 2))
    return f'max diff {max_diff:.2e}, {snr:.1f} dB', snr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('reference')
    parser.add_argument('test')
    parser.add_argument('--min-db', type=float, help='Exit with status 1 if any stem is below this many dB')
    args = parser.parse_args()

    names = sorted(n for n in os.listdir(args.reference) if n.endswith('.wav'))
    if not names:
        sys.exit(f'No WAV files in {args.reference}')
    worst = float('inf')
    for name in names:
        test_path = os.path.join(args.test, name)
        if not os.path.isfile(test_path):
            print(f'{name}: missing from {args.test}')
            worst = -float('inf')
            continue
        text, snr = compare(os.path.join(args.reference, name), test_path)
        worst = min(worst, snr if snr is not None else -float('inf'))
        print(f'{name}: {text}')
    if args.min_db is not None and worst < args.min_db:
        sys.exit(1)


if __name__ == '__main__':
    main()

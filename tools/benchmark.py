"""Time this fork against upstream UVR on one song and check the output.

Runs tools/headless_separate.py for:
  upstream            unmodified UVR, full precision
  fork                this fork, full precision (should match upstream)
  fork half           this fork with Half Precision on
  fork half batch N   same, with a bigger batch (only if --batch is given)
then prints time, peak VRAM and how far each output is from upstream's.

Example (Roformer on an NVIDIA GPU):
    git clone -b v5.6.0_roformer_add+directml https://github.com/Anjok07/ultimatevocalremovergui ../uvr-upstream
    python tools/benchmark.py --upstream-dir ../uvr-upstream --arch roformer \
        --model models/MDX_Net_Models/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
        --config models/MDX_Net_Models/model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml \
        --input song.flac --device cuda --batch 4
"""
import argparse
import json
import os
import subprocess
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
FORK_DIR = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)
from compare_outputs import compare  # noqa: E402


def run(name, uvr_dir, args, out_root, settings):
    out_dir = os.path.join(out_root, name.replace(' ', '_'))
    cmd = [sys.executable, os.path.join(TOOLS_DIR, 'headless_separate.py'), '--uvr-dir', uvr_dir,
           '--arch', args.arch, '--model', args.model, '--input', args.input, '--out', out_dir,
           '--device', args.device, '--repeat', str(args.repeat), '--quiet']
    if args.config:
        cmd += ['--config', args.config]
    for model_data in args.model_data or []:
        cmd += ['--model-data', model_data]
    if settings:
        cmd += ['--set', *settings]
    print(f'Running {name}...', flush=True)
    subprocess.run(cmd, check=True)
    with open(os.path.join(out_dir, 'timings.json')) as f:
        return out_dir, json.load(f)['runs']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--upstream-dir', required=True, help='Checkout of unmodified upstream UVR')
    parser.add_argument('--arch', required=True, choices=['roformer', 'mdxc', 'mdx', 'vr', 'demucs'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--config')
    parser.add_argument('--model-data', action='append')
    parser.add_argument('--input', required=True)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'mps', 'directml'])
    parser.add_argument('--batch', type=int, help='Also run the fork in half precision with this batch size')
    parser.add_argument('--repeat', type=int, default=2, help='Separations per run; the second shows the model cache')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='Settings applied to every run')
    parser.add_argument('--out', default='benchmark_out')
    args = parser.parse_args()
    for attr in ('model', 'config', 'input'):
        if getattr(args, attr):
            setattr(args, attr, os.path.abspath(getattr(args, attr)))
    if args.model_data:
        args.model_data = [os.path.abspath(p) for p in args.model_data]
    out_root = os.path.abspath(args.out)

    runs = [('upstream', os.path.abspath(args.upstream_dir), args.set),
            ('fork', FORK_DIR, args.set),
            ('fork half', FORK_DIR, args.set + ['is_half_precision=True'])]
    if args.batch:
        batch_key = {'vr': 'batch_size', 'demucs': 'demucs_batch_size'}.get(args.arch, 'mdx_batch_size')
        runs.append((f'fork half batch {args.batch}', FORK_DIR, args.set + ['is_half_precision=True', f'{batch_key}={args.batch}']))

    results = [(name, *run(name, uvr_dir, args, out_root, settings)) for name, uvr_dir, settings in runs]

    reference_dir = results[0][1]
    stems = sorted(n for n in os.listdir(reference_dir) if n.endswith('.wav'))
    print()
    print(f"{'run':<22}{'1st song':>10}{'2nd song':>10}{'peak VRAM':>11}   worst stem vs upstream")
    for name, out_dir, timings in results:
        first = f"{timings[0]['seconds']:.1f}s"
        second = f"{timings[1]['seconds']:.1f}s" if len(timings) > 1 else '-'
        vram = f"{max(t.get('peak_vram_mb', 0) for t in timings)} MB" if 'peak_vram_mb' in timings[0] else '-'
        if out_dir == reference_dir:
            diff = '(reference)'
        else:
            scored = [compare(os.path.join(reference_dir, s), os.path.join(out_dir, s)) for s in stems]
            diff = min(scored, key=lambda r: r[1] if r[1] is not None else -1)[0]
        print(f'{name:<22}{first:>10}{second:>10}{vram:>11}   {diff}')
    print('\nPeak VRAM counts PyTorch memory only (not ONNX Runtime). "identical" means bit-exact;')
    print('above ~100 dB is rounding noise; half precision measured 60-80 dB in testing.')
    if args.arch == 'demucs' and args.batch:
        print('Demucs with Shifts above 0 uses random offsets, and batching changes which ones are drawn,')
        print('so the batch row differs from upstream by design. Add --set shifts=0 to check batching exactly.')


if __name__ == '__main__':
    main()

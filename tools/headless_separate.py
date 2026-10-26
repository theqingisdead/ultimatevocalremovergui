"""Run a UVR separation without the GUI, for benchmarking and regression checks.

Works against this fork or an unmodified upstream UVR checkout (pass --uvr-dir),
so the same command can time both and save their outputs for compare_outputs.py.

Example:
    python tools/headless_separate.py --uvr-dir . --arch roformer \
        --model models/MDX_Net_Models/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
        --input song.wav --out out/fork --device cuda --repeat 2
"""
import argparse
import ast
import hashlib
import json
import os
import random
import sys
import time
import types


def parse_settings(pairs):
    settings = {}
    for pair in pairs or []:
        key, _, value = pair.partition('=')
        try:
            settings[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            settings[key] = value
    return settings


def model_hash(path):
    # Same scheme as ModelData.get_model_hash in UVR.py
    with open(path, 'rb') as f:
        try:
            f.seek(-10000 * 1024, 2)
        except OSError:
            f.seek(0)
        return hashlib.md5(f.read()).hexdigest()


def find_model_params(path, json_paths):
    digest = model_hash(path)
    for json_path in json_paths:
        if not os.path.isfile(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)
        if digest in data:
            return data[digest]
    hash_jsons = [os.path.join(os.path.dirname(p), f'{digest}.json') for p in json_paths]
    for hash_json in hash_jsons:
        if os.path.isfile(hash_json):
            with open(hash_json) as f:
                return json.load(f)
    raise SystemExit(f'No UVR parameters found for {path} (hash {digest}). Pass --model-data with a model_data json that has it.')


def import_uvr(uvr_dir):
    uvr_dir = os.path.abspath(uvr_dir)
    os.chdir(uvr_dir)
    sys.path.insert(0, uvr_dir)

    try:
        import torch_directml  # noqa: F401
    except ImportError:
        # Upstream separate.py imports torch_directml unconditionally outside macOS.
        stub = types.ModuleType('torch_directml')
        stub.is_available = lambda: False
        stub.device = lambda *a, **k: 'cpu'
        sys.modules['torch_directml'] = stub

    try:
        import tkinter  # noqa: F401
    except ImportError:
        # Upstream demucs/apply.py and demucs/utils.py import tkinter without using it.
        sys.modules['tkinter'] = types.ModuleType('tkinter')

    import separate
    return separate


def make_model_data(constants, arch, model_path, settings, config_path, model_data_jsons, device):
    c = constants
    s = {
        # GUI defaults from DEFAULT_DATA in gui_data/constants.py
        'aggression_setting': 5, 'window_size': 512, 'batch_size': 1, 'is_tta': False,
        'is_post_process': False, 'is_high_end_process': False, 'post_process_threshold': 0.2,
        'mdx_segment_size': 256, 'mdx_batch_size': 1, 'overlap_mdx': c.DEFAULT, 'overlap_mdx23': 8,
        'is_mdx_c_seg_def': True, 'margin': 44100, 'compensate': c.AUTO_SELECT,
        'segment': c.DEF_OPT, 'overlap': 0.25, 'shifts': 2, 'demucs_batch_size': 1, 'is_split_mode': True, 'is_chunk_demucs': False,
        'demucs_stems': c.ALL_STEMS, 'mdx_stems': c.ALL_STEMS, 'is_demucs_combine_stems': True,
        'is_mdx23_combine_stems': True, 'is_match_frequency_pitch': True, 'semitone_shift': 0,
        'is_invert_spec': False, 'is_normalization': False, 'wav_type_set': 'FLOAT',
        'save_format': c.WAV, 'mp3_bit_set': '320k', 'device_set': c.DEFAULT, 'is_denoise': False,
        'is_denoise_model': False, 'is_primary_stem_only': False, 'is_secondary_stem_only': False,
        'is_half_precision': False,
    }
    s.update(settings)

    md = types.SimpleNamespace()
    md.DENOISER_MODEL = s.get('denoiser_model')
    md.DEVERBER_MODEL = s.get('deverber_model')
    md.is_deverb_vocals = False
    md.deverb_vocal_opt = c.DEVERB_MAPPER['Main Vocals Only']
    md.is_denoise_model = s['is_denoise_model']
    md.is_gpu_conversion = -1 if device == 'cpu' else 0
    md.is_normalization = s['is_normalization']
    md.is_half_precision = s['is_half_precision']
    md.is_use_directml = device == 'directml'
    md.is_primary_stem_only = s['is_primary_stem_only']
    md.is_secondary_stem_only = s['is_secondary_stem_only']
    md.is_denoise = s['is_denoise']
    md.is_mdx_c_seg_def = s['is_mdx_c_seg_def']
    md.mdx_batch_size = s['mdx_batch_size']
    md.mdxnet_stem_select = s['mdx_stems']
    md.overlap = s['overlap']
    md.overlap_mdx = s['overlap_mdx']
    md.overlap_mdx23 = int(s['overlap_mdx23'])
    md.semitone_shift = float(s['semitone_shift'])
    md.is_pitch_change = md.semitone_shift != 0
    md.is_match_frequency_pitch = s['is_match_frequency_pitch']
    md.is_mdx_ckpt = False
    md.is_mdx_c = False
    md.is_mdx_combine_stems = s['is_mdx23_combine_stems']
    md.mdx_c_configs = None
    md.mdx_dim_f_set = md.mdx_dim_t_set = md.mdx_n_fft_scale_set = None
    md.compensate = None
    md.wav_type_set = s['wav_type_set']
    md.device_set = s['device_set']
    md.mp3_bit_set = s['mp3_bit_set']
    md.save_format = s['save_format']
    md.is_invert_spec = s['is_invert_spec']
    md.is_mixer_mode = False
    md.demucs_stems = s['demucs_stems']
    md.is_demucs_combine_stems = s['is_demucs_combine_stems']
    md.demucs_source_list = []
    md.demucs_source_map = {}
    md.demucs_stem_count = 0
    md.mixer_path = None
    md.model_path = os.path.abspath(model_path)
    md.model_name = os.path.splitext(os.path.basename(model_path))[0]
    md.model_basename = md.model_name
    md.primary_stem = md.secondary_stem = md.primary_stem_native = None
    md.is_ensemble_mode = False
    md.ensemble_primary_stem = md.ensemble_secondary_stem = None
    md.primary_model_primary_stem = None
    md.is_secondary_model = False
    md.secondary_model = None
    md.secondary_model_scale = None
    md.pre_proc_model = None
    md.is_pre_proc_model = False
    md.model_samplerate = 44100
    md.model_capacity = 32, 128
    md.is_vr_51_model = False
    md.is_demucs_pre_proc_model_inst_mix = False
    md.secondary_model_4_stem = []
    md.secondary_model_4_stem_scale = []
    md.is_multi_stem_ensemble = False
    md.is_karaoke = False
    md.is_bv_model = False
    md.bv_model_rebalance = 0
    md.is_sec_bv_rebalance = False
    md.is_secondary_model_activated = False
    md.vocal_split_model = None
    md.is_vocal_split_model = False
    md.is_save_inst_vocal_splitter = False
    md.is_inst_only_voc_splitter = False
    md.is_save_vocal_only = False
    md.is_roformer = False
    md.is_target_instrument = False
    md.is_primary_model_primary_stem_only = False
    md.is_primary_model_secondary_stem_only = False

    if arch == 'vr':
        from lib_v5.vr_network.model_param_init import ModelParameters
        params = find_model_params(model_path, model_data_jsons or [os.path.join('models', 'VR_Models', 'model_data', 'model_data.json')])
        md.process_method = c.VR_ARCH_TYPE
        md.aggression_setting = float(int(s['aggression_setting']) / 100)
        md.is_tta = s['is_tta']
        md.is_post_process = s['is_post_process']
        md.window_size = int(s['window_size'])
        md.batch_size = int(s['batch_size'])
        md.crop_size = 256
        md.is_high_end_process = 'mirroring' if s['is_high_end_process'] else 'None'
        md.post_process_threshold = float(s['post_process_threshold'])
        md.vr_model_param = ModelParameters(os.path.join('lib_v5', 'vr_network', 'modelparams', f"{params['vr_model_param']}.json"))
        md.model_samplerate = md.vr_model_param.param['sr']
        md.primary_stem = md.primary_stem_native = params['primary_stem']
        if 'nout' in params and 'nout_lstm' in params:
            md.model_capacity = params['nout'], params['nout_lstm']
            md.is_vr_51_model = True
    elif arch in ('mdx', 'mdxc', 'roformer'):
        md.process_method = c.MDX_ARCH_TYPE
        md.margin = int(s['margin'])
        md.chunks = 0
        md.mdx_segment_size = int(s['mdx_segment_size'])
        if arch == 'mdx':
            params = find_model_params(model_path, model_data_jsons or [os.path.join('models', 'MDX_Net_Models', 'model_data', 'model_data.json')])
            md.compensate = params['compensate'] if s['compensate'] == c.AUTO_SELECT else float(s['compensate'])
            md.mdx_dim_f_set = params['mdx_dim_f_set']
            md.mdx_dim_t_set = params['mdx_dim_t_set']
            md.mdx_n_fft_scale_set = params['mdx_n_fft_scale_set']
            md.primary_stem = md.primary_stem_native = params['primary_stem']
        else:
            import yaml
            from ml_collections import ConfigDict
            with open(config_path) as f:
                md.mdx_c_configs = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            md.is_mdx_c = True
            md.is_roformer = arch == 'roformer'
            target = md.mdx_c_configs.training.target_instrument
            if target:
                md.is_target_instrument = True
                md.primary_stem = target
            else:
                instruments = md.mdx_c_configs.training.instruments
                md.primary_stem = instruments[0] if len(instruments) == 2 else md.mdxnet_stem_select
    elif arch == 'demucs':
        md.process_method = c.DEMUCS_ARCH_TYPE
        md.shifts = int(s['shifts'])
        md.demucs_batch_size = int(s['demucs_batch_size'])
        md.is_split_mode = s['is_split_mode']
        md.segment = s['segment']
        md.is_chunk_demucs = s['is_chunk_demucs']
        md.demucs_version = c.DEMUCS_V4
        md.demucs_source_list, md.demucs_source_map, md.demucs_stem_count = c.DEMUCS_4_SOURCE, c.DEMUCS_4_SOURCE_MAPPER, 4
        md.primary_stem = c.PRIMARY_STEM if md.demucs_stems == c.ALL_STEMS else md.demucs_stems
    else:
        raise SystemExit(f'Unknown arch {arch}')

    md.secondary_stem = c.secondary_stem(md.primary_stem)
    return md


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--uvr-dir', default='.', help='UVR checkout to run (this fork or upstream)')
    parser.add_argument('--arch', required=True, choices=['roformer', 'mdxc', 'mdx', 'vr', 'demucs'])
    parser.add_argument('--model', required=True, help='Model file (.ckpt/.onnx/.pth, or .yaml for Demucs v4)')
    parser.add_argument('--config', help='Model config .yaml (Roformer / MDX23C)')
    parser.add_argument('--model-data', action='append', help="UVR model_data json to look up the model's parameters")
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps', 'directml'])
    parser.add_argument('--repeat', type=int, default=1, help='Separate the input N times in one process (later runs show model caching)')
    parser.add_argument('--set', nargs='*', metavar='KEY=VALUE', help='Override a GUI setting, e.g. overlap_mdx23=4 mdx_batch_size=4')
    parser.add_argument('--seed', type=int, default=0, help='Seed for Demucs random shifts, so runs are comparable')
    parser.add_argument('--emulate-half-on-cpu', action='store_true',
                        help='Testing only: run the fork\'s half precision path with CPU autocast to estimate its output difference without a GPU (slow)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    config_path = os.path.abspath(args.config) if args.config else None
    input_path = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out)
    model_data_jsons = [os.path.abspath(p) for p in args.model_data] if args.model_data else None
    os.makedirs(out_dir, exist_ok=True)

    separate = import_uvr(args.uvr_dir)
    import numpy as np
    import torch
    import gui_data.constants as constants

    settings = parse_settings(args.set)
    model_data = make_model_data(constants, args.arch, model_path, settings, config_path, model_data_jsons, args.device)
    if args.emulate_half_on_cpu:
        model_data.is_half_precision = True
        original_init = separate.SeperateAttributes.__init__

        def init_with_cpu_autocast(self, *a, **k):
            original_init(self, *a, **k)
            self.autocast_device = 'cpu'

        separate.SeperateAttributes.__init__ = init_with_cpu_autocast

    classes = {constants.VR_ARCH_TYPE: separate.SeperateVR, constants.DEMUCS_ARCH_TYPE: separate.SeperateDemucs}
    cls = classes.get(model_data.process_method) or (separate.SeperateMDXC if model_data.is_mdx_c else separate.SeperateMDX)

    def write_to_console(text, base_text=''):
        if not args.quiet:
            print(f'{base_text}{text}', end='' if text.endswith('\n') else '\n', flush=True)

    audio_file_base = os.path.splitext(os.path.basename(input_path))[0]

    results = []
    for run in range(args.repeat):
        # UVR writes "<name>_<timestamp>.wav" instead of overwriting, so clear earlier stems first
        for name in os.listdir(out_dir):
            if name.startswith(f'{audio_file_base}_(') and name.endswith('.wav'):
                os.remove(os.path.join(out_dir, name))
        process_data = {
            'model_data': model_data,
            'export_path': out_dir,
            'audio_file_base': audio_file_base,
            'audio_file': input_path,
            'set_progress_bar': lambda step, inference_iterations=0: None,
            'write_to_console': write_to_console,
            'process_iteration': lambda: None,
            'cached_source_callback': lambda process_method, model_name=None: (None, None),
            'cached_model_source_holder': lambda *a, **k: None,
            'list_all_models': [model_data.model_basename],
            'is_ensemble_master': False,
            'is_4_stem_ensemble': False,
        }
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        cls(model_data, process_data).seperate()
        if args.device == 'cuda':
            torch.cuda.synchronize()
        result = {'run': run + 1, 'seconds': round(time.perf_counter() - start, 3)}
        if args.device == 'cuda':
            result['peak_vram_mb'] = round(torch.cuda.max_memory_allocated() / 2**20)
        results.append(result)
        print(json.dumps(result), flush=True)

    with open(os.path.join(out_dir, 'timings.json'), 'w') as f:
        json.dump({'arch': args.arch, 'model': os.path.basename(model_path), 'device': args.device,
                   'settings': settings, 'runs': results}, f, indent=2)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run all eight numbered stages, stopping before a dependent stage on failure."""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from offline_qa.common import load_config, preflight_services, prepare_media


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--config', default=str(package / 'config.example.json'))
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--url-map')
    parser.add_argument('--limit', type=int, help='Pilot on the first N source rows')
    parser.add_argument('--supplement-from', help='Prior step-7 records; refill rejected pools in a NEW run directory')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='Validate local inputs and service settings; make no API calls')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    # Fail before generation if a later paid stage is missing a service/model setting.
    config = load_config(args.config)
    preflight_services(config)
    if args.preflight:
        rows, _ = prepare_media(Path(args.input).resolve(), Path(args.url_map).resolve() if args.url_map else None,
                                args.limit, config.get('video_input', 'url'))
        print(f'Preflight passed: {len(rows)} unique media; local inputs and service settings only. No API calls.')
        return
    names = ['prepare', 'generate', 'validate', 'blind_filter', 'verify', 'calibrate', 'sample', 'export']
    for stage, name in enumerate(names, 1):
        cmd = [sys.executable, str(package / f'{stage}_{name}.py'), '--config', args.config,
               '--run-dir', args.run_dir]
        if args.resume:
            cmd.append('--resume')
        if stage == 1:
            cmd.extend(['--input', args.input])
            if args.url_map:
                cmd.extend(['--url-map', args.url_map])
            if args.limit is not None:
                cmd.extend(['--limit', str(args.limit)])
        if stage == 2 and args.supplement_from:
            cmd.extend(['--supplement-from', args.supplement_from])
        result = subprocess.run(cmd)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

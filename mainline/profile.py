#!/usr/bin/env python3
"""Render a two-rank test plan and verify new-format assets; never execute a launch."""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_config(path):
    c = json.loads(path.read_text())
    for key, expected in [('context', 524288), ('pool', 2621440), ('parallel', 4),
                          ('max_tokens', 32768), ('backend_port', 18036), ('model_name', 'spark')]:
        if c.get(key) != expected:
            raise ValueError('This preserved-settings profile requires ' + key + '=' + str(expected))
    ipaddress.IPv4Address(c['master'])
    if not 1024 <= c['master_port'] <= 65535:
        raise ValueError('Invalid rank rendezvous port')
    for key in ['checkpoint', 'assets']:
        if not Path(c[key]).is_absolute():
            raise ValueError(key + ' must be an absolute path')
    for key in ['engram', 'prepared']:
        if len(c[key]) != 2 or any(not Path(p).is_absolute() for p in c[key]):
            raise ValueError(key + ' requires two absolute rank paths')
    if any(p in [c['checkpoint'], c['assets'], *c['engram']] for p in c['prepared']):
        raise ValueError('Prepared weights need separate new namespaces')
    return c


def environment(c, rank):
    return {'TF_EXPERIMENTAL': 'deepseek', 'TF_TP_WORLD': '2', 'TF_TP_RANK': str(rank),
            'TF_TP_DEVICE': '0', 'TF_COMM_BACKEND': 'roce', 'TF_TP_PLAN_LINK': 'rdma', 'TF_TP_MASTER': c['master'], 'TF_TP_PORT': str(c['master_port']),
            'TF_DSV41_ASSETS': '/assets', 'TF_DSV41_CONTEXT': str(c['context']),
            'TF_DSV41_POOL_TOKENS': str(c['pool']), 'TF_DSV41_SLOTS': str(c['parallel']),
            'TF_DSV41_SLOT_DRAFTS': '1', 'TF_DSV41_DRAFTS': '1',
            'TF_DSV41_ENGRAM_DIR': '/engram', 'TF_DSV41_PREPARED': '/prepared',
            'TF_DSV41_PREPARED_WRITE': '1', 'TF_DSV41_PREPARED_VERIFY': '1',
            'TF_DSV41_PREFILL': 'full', 'TF_DSV41_OWN_PREFILL': '1',
            'TF_DSV41_PROMPT_TAIL': 'verify', 'TF_DSV41_IMAGES': '0',
            'TF_DSV41_PREFILL_PIECES': '1', 'TF_DSV41_SESSION_SLOTS': '1',
            'TF_DSV41_SESSIONS': '1', 'TF_DSV41_SESSION_RAM_MIB': '128',
            'TF_DSV41_FLOOR_GIB': '3.5', 'TF_DSV41_FLOOR_HARD_GIB': '2.5'}


def activation_blockers(c):
    blockers = ['GPU correctness, memory, performance and continuation gates have not run',
                'Native checkpoint, RoPE, token-map, Engram and AOT assets require validation',
                'Text-only candidate: image input is not qualified',
                'Equivalent graph-cache cap and RDMA behavior require validation',
                'Candidate-scoped 2.5 GiB soft / 1 GiB hard host memory guards must be installed before activation']
    if c.get('disk_cache', {}).get('enabled'):
        blockers.append('Disk caching requires porting the 24-hour expiry and 3.5 GiB read/write admission policy; legacy spill-v2 is not mounted')
    return blockers


def commands(c):
    rows = []
    for rank in (0, 1):
        name = 'dsv41-mainline-2ba55ec7-rank' + str(rank)
        cmd = ['docker', 'run', '-d', '--name', name, '--network', 'host', '--ipc=host',
               '--gpus', 'all', '--ulimit', 'memlock=-1', '--cap-add', 'IPC_LOCK',
               '--device', '/dev/infiniband', '--label', 'tf.engine=mainline',
               '-v', c['checkpoint'] + ':/model:ro', '-v', c['assets'] + ':/assets:ro',
               '-v', c['engram'][rank] + ':/engram:ro', '-v', c['prepared'][rank] + ':/prepared']
        for k, v in sorted(environment(c, rank).items()):
            cmd += ['-e', k + '=' + v]
        cmd += [c['image']]
        if rank == 0:
            cmd += ['tensorfold-dsv41', 'serve', '/model', '--backend', 'cuda',
                    '--host', '127.0.0.1', '--port', str(c['backend_port']), '--name', 'spark',
                    '--context', str(c['context']), '--parallel', str(c['parallel']),
                    '--max-tokens', str(c['max_tokens']), '--temperature', '1.0',
                    '--top-p', '0.95', '--top-k', '20', '--thinking', '--loop-guard']
        else:
            cmd += ['tf-dsv41', 'follow', '/model', '/assets']
        rows.append({'rank': rank, 'argv': cmd})
    return rows


def verify_assets(c):
    root = Path(c['assets'])
    receipt = json.loads((root / 'manifest.json').read_text())
    pins = json.loads((ROOT / 'pins.json').read_text())
    if receipt['tensorfold'] != pins['tensorfold']['revision'] or receipt['patches'] != [p['sha256'] for p in pins['patches']]:
        raise ValueError('Assets belong to another engine or patch set')
    want = digest(json.dumps([environment(c, r) for r in (0, 1)], sort_keys=True).encode())
    if receipt['profile_sha256'] != want:
        raise ValueError('Asset planning settings differ')
    for name in ['config.json', 'tokenizer.json']:
        if digest((Path(c['checkpoint']) / name).read_bytes()) != receipt['model_files'][name]:
            raise ValueError('Checkpoint/tokenizer differs: ' + name)
    needed = {'rope.bin', 'engram-token-map.bin', 'aot/aot.json'}
    if not needed <= receipt['files'].keys():
        raise ValueError('Asset manifest is incomplete')
    for name, expected in receipt['files'].items():
        rel = Path(name)
        path = root / rel
        if rel.is_absolute() or '..' in rel.parts or root.is_symlink() or any((root / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)):
            raise ValueError('Unsafe asset path')
        if digest(path.read_bytes()) != expected:
            raise ValueError('Asset digest differs: ' + name)
    rope = (root / 'rope.bin').read_bytes()
    if len(rope) < 16 or rope[:8] != b'DSV41RP1':
        raise ValueError('Legacy or invalid RoPE format')
    rows, cols = struct.unpack('<II', rope[8:16])
    if rows < c['context'] + 2048 or cols != 64 or len(rope) != 16 + rows * cols * 8:
        raise ValueError('RoPE coverage or length differs')
    if (root / 'engram-token-map.bin').read_bytes()[:8] != b'DSENGR01':
        raise ValueError('Legacy token map is not a mainline serving asset')
    aot = json.loads((root / 'aot/aot.json').read_text())
    if not aot.get('kernels'):
        raise ValueError('Empty AOT kernel set')
    return {'asset_digests_verified': True, 'gpu_qualification': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['plan', 'verify-assets'])
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = read_config(args.config)
    if args.operation == 'verify-assets':
        out = verify_assets(c)
    else:
        out = {'executed': False, 'qualified': False, 'rank_start_order': [1, 0],
               'activation_blockers': activation_blockers(c), 'commands': commands(c),
               'aot_environment': [environment(c, r) for r in (0, 1)],
               'profile_sha256': digest(json.dumps([environment(c, r) for r in (0, 1)], sort_keys=True).encode())}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

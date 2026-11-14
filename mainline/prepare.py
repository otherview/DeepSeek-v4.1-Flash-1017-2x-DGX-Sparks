#!/usr/bin/env python3
"""Reproducible mainline source and CPU-build preparation. Never operates a Spark or a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
IGNORE = {'.git', '.zig-cache', '.global-cache', 'zig-out'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pins():
    return json.loads((ROOT / 'pins.json').read_text())


def manifest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and not any(x in IGNORE for x in p.relative_to(root).parts)}


def extract(archive, root):
    with tarfile.open(archive) as source:
        for member in source:
            rel = Path(*Path(member.name).parts[1:])
            if not rel.parts:
                continue
            if rel.is_absolute() or '..' in rel.parts or member.issym() or member.islnk():
                raise RuntimeError('Unsafe archive member')
            out = root / rel
            if member.isdir():
                out.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(source.extractfile(member).read())
                out.chmod(member.mode & 0o777)
            else:
                raise RuntimeError('Unsupported archive member')


def source():
    spec = pins()
    DATA.mkdir(exist_ok=True)
    archive = DATA / 'tensorfold-main.tar.gz'
    if not archive.exists():
        part = archive.with_suffix('.part')
        urllib.request.urlretrieve(spec['tensorfold']['url'], part)
        if sha(part) != spec['tensorfold']['sha256']:
            raise RuntimeError('TensorFold archive digest differs')
        part.rename(archive)
    if sha(archive) != spec['tensorfold']['sha256']:
        raise RuntimeError('TensorFold archive digest differs')
    for patch in spec['patches']:
        if sha(ROOT / patch['file']) != patch['sha256']:
            raise RuntimeError('Pinned patch digest differs: ' + patch['file'])
    with tempfile.TemporaryDirectory(prefix='mainline-', dir=DATA) as tmp:
        tree = Path(tmp) / 'source'
        extract(archive, tree)
        subprocess.run(['git', 'init', '-q', str(tree)], check=True)
        for patch in spec['patches']:
            path = str(ROOT / patch['file'])
            subprocess.run(['git', 'apply', '--check', path], cwd=tree, check=True)
            subprocess.run(['git', 'apply', path], cwd=tree, check=True)
        files = manifest(tree)
        dest = DATA / 'source'
        if dest.exists():
            if manifest(dest) != files:
                raise RuntimeError('Existing source differs; it was left unchanged')
        else:
            os.rename(tree, dest)
        (DATA / 'source-manifest.json').write_text(json.dumps(files, sort_keys=True) + '\n')
    return dest


def host_build(zig):
    tree = source()
    spec = pins()
    if sha(zig / 'zig') != spec['compiler']['binary_sha256']:
        raise RuntimeError('Use the pinned Linux ARM64 compiler')
    args = ['docker', 'run', '--rm', '--runtime', 'runc', '--network', 'none',
            '--cpus', '2', '--memory', '3g', '-v', str(tree) + ':/source',
            '-v', str(zig.resolve()) + ':/toolchain:ro', '-w', '/source',
            '-e', 'ZIG_GLOBAL_CACHE_DIR=/source/.global-cache',
            '-e', 'ZIG_LOCAL_CACHE_DIR=/source/.zig-cache', spec['cpu_builder'],
            '/toolchain/zig', 'build', '-j2', 'dsv41-assets',
            'test-dsv41-assets-host', 'test-dsv41-token-map', 'test-dsv41-draft',
            'test-dsv41-forward', '--summary', 'all']
    started = time.time()
    with (DATA / 'host-build.log').open('w') as output:
        result = subprocess.run(args, stdout=output, stderr=subprocess.STDOUT)
    receipt = {'exit_code': result.returncode, 'seconds': round(time.time() - started, 2),
               'source_manifest_sha256': sha(DATA / 'source-manifest.json'),
               'gpu_access': False, 'model_access': False,
               'production_operations': [], 'log': 'data/host-build.log'}
    (DATA / 'host-build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError('Host build failed; see data/host-build.log')
    print(json.dumps(receipt))


def context():
    tree = source()
    dest = DATA / 'context'
    if dest.exists():
        if (manifest(dest / 'source') != manifest(tree) or
            any((dest / n).read_bytes() != (ROOT / n).read_bytes() for n in ['Dockerfile', 'pins.json'])):
            raise RuntimeError('Existing build context differs; it was left unchanged')
        print(json.dumps({'context': str(dest), 'reused': True, 'production_operations': []}))
        return
    dest.mkdir()
    shutil.copytree(tree, dest / 'source', ignore=shutil.ignore_patterns(*IGNORE))
    shutil.copy2(ROOT / 'Dockerfile', dest / 'Dockerfile')
    shutil.copy2(ROOT / 'pins.json', dest / 'pins.json')
    for name in ['LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md']:
        if (tree / name).exists():
            shutil.copy2(tree / name, dest / name)
    print(json.dumps({'context': str(dest), 'source_manifest_sha256': sha(DATA / 'source-manifest.json'),
                      'build_executed': False, 'production_operations': []}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['source', 'host-build', 'context'])
    parser.add_argument('--zig', type=Path)
    args = parser.parse_args()
    if args.operation == 'host-build':
        if args.zig is None:
            parser.error('--zig is required')
        host_build(args.zig)
    elif args.operation == 'context':
        context()
    else:
        tree = source()
        print(json.dumps({'source': str(tree), 'files': len(manifest(tree)),
                          'production_operations': []}))


if __name__ == '__main__':
    main()

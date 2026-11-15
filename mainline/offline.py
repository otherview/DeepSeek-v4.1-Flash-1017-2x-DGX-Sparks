#!/usr/bin/env python3
"""Local dependency/image packaging only. Never SSHs, loads a model or operates a service."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import prepare
import profile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def image_identity(info):
    return {'architecture': info['Architecture'], 'rootfs_layers': info['RootFS']['Layers'],
            'runtime_config_sha256': hashlib.sha256(json.dumps(info['Config'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def fetch(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_suffix('.part')
        urllib.request.urlretrieve(url, part)
        if sha(part) != expected:
            raise RuntimeError('Download digest differs: ' + path.name)
        part.rename(path)
    if sha(path) != expected:
        raise RuntimeError('Existing dependency differs: ' + path.name)


def generator():
    pins = prepare.pins()
    spec = pins['reference']
    cache = DATA / 'reference'
    archive = cache / 'source.tar.gz'
    fetch('https://codeload.github.com/' + spec['repository'] + '/tar.gz/' + spec['revision'], archive, spec['archive_sha256'])
    py = spec['python_source']
    pyarchive = cache / 'python-source.tar.gz'
    fetch('https://codeload.github.com/' + py['repository'] + '/tar.gz/' + py['revision'], pyarchive, py['archive_sha256'])
    for name, expected in py['patches'].items():
        fetch('https://raw.githubusercontent.com/' + py['patch_repository'] + '/' + py['patch_revision'] + '/patches/' + name, cache / name, expected)
    with tempfile.TemporaryDirectory(dir=DATA, prefix='generator-') as tmp:
        tree = Path(tmp) / 'context'
        tree.mkdir()
        # The recipe archive contains unrelated benchmark symlinks. Select only
        # the regular engine files and top-level notices; never follow a link.
        with tarfile.open(archive) as tar:
            for member in tar:
                rel = Path(*Path(member.name).parts[1:])
                if not rel.parts or (rel.parts[0] != 'engine' and rel.name not in ['LICENSE', 'NOTICE', 'ATTRIBUTION.md', 'THIRD_PARTY_NOTICES.md']):
                    continue
                if rel.is_absolute() or '..' in rel.parts or not (member.isdir() or member.isfile()):
                    raise RuntimeError('Unsafe generator archive member')
                out = tree / 'reference-tools' / rel
                if member.isdir():
                    out.mkdir(parents=True, exist_ok=True)
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, out.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    out.chmod(member.mode & 0o777)
        reference = tree / 'python-reference'
        prepare.extract(pyarchive, reference)
        subprocess.run(['git', 'init', '-q', str(reference)], check=True)
        for name in py['patches']:
            subprocess.run(['git', 'apply', '--check', str((cache / name).resolve())], cwd=reference, check=True)
            subprocess.run(['git', 'apply', '--whitespace=nowarn', str((cache / name).resolve())], cwd=reference, check=True)
        shutil.rmtree(reference / '.git')
        if not (reference / 'src/tensorfold/families/deepseek_v41/cuda/rope.py').is_file():
            raise RuntimeError('Python reference reconstruction is incomplete')
        (tree / 'pins.json').write_bytes((ROOT / 'pins.json').read_bytes())
        (tree / 'Dockerfile').write_bytes((ROOT / 'Dockerfile.assets').read_bytes())
        dest = DATA / 'generator-context'
        if dest.exists():
            if prepare.manifest(tree) != prepare.manifest(dest):
                raise RuntimeError('Existing generator context differs; left unchanged')
        else:
            os.rename(tree, dest)
    print(json.dumps({'context': str(dest), 'production_operations': [], 'model_access': False}))


def build_generator(image):
    generator()
    validation = json.loads((ROOT / 'VALIDATION.json').read_text())
    base = json.loads(subprocess.check_output(['docker', 'image', 'inspect', 'tensorfold-dsv41:mainline-2ba55ec7']))[0]
    if base['Id'] != validation['native_image']['id'] or base['Architecture'] != 'arm64':
        raise RuntimeError('Local native base differs from the verified build receipt')
    subprocess.run(['docker', 'build', '--network', 'none', '--pull=false', '--platform', 'linux/arm64',
                    '-t', image, str(DATA / 'generator-context')], check=True)


def package(image, output):
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'image.json'
    archive = output / 'candidate-image.tar.gz'
    info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]
    if info['Architecture'] != 'arm64' or info['Config']['Labels'].get('tf.ref') != prepare.pins()['tensorfold']['revision']:
        raise RuntimeError('Unexpected candidate image identity')
    if receipt.exists():
        record = verify_package(output)
        if any(record[k] != v for k, v in image_identity(info).items()):
            raise RuntimeError('Existing package belongs to another image')
        print(json.dumps(record, indent=2))
        return
    if archive.exists():
        raise RuntimeError('Unreceipted archive exists; choose a new output directory')
    part = output / 'candidate-image.tar.gz.part'
    with subprocess.Popen(['docker', 'image', 'save', image], stdout=subprocess.PIPE) as proc:
        with part.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as zipped:
            shutil.copyfileobj(proc.stdout, zipped, 4 * 1024 * 1024)
        if proc.wait():
            raise RuntimeError('Docker export failed; partial archive retained')
    part.rename(archive)
    record = {'image': image, 'local_image_id': info['Id'], 'architecture': info['Architecture'],
              'tensorfold': prepare.pins()['tensorfold']['revision'],
              'archive': archive.name, 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
              'rootfs_layers': info['RootFS']['Layers'],
              'production_operations': [], 'gpu_inference_performed': False,
              'scope': 'Complete Docker save archive; no existing parent image required. Image IDs can differ between Docker storage backends; compare ordered RootFS layers and runtime config after loading.'}
    # Bind runtime config as well as layers, without publishing environment values.
    record['runtime_config_sha256'] = hashlib.sha256(json.dumps(info['Config'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


def verify_package(output):
    record = json.loads((output / 'image.json').read_text())
    if record['archive'] != 'candidate-image.tar.gz':
        raise RuntimeError('Unexpected archive name')
    archive = output / record['archive']
    if archive.stat().st_size != record['archive_bytes'] or sha(archive) != record['archive_sha256']:
        raise RuntimeError('Packaged image digest/size differs')
    return record


def inspect_archive(output):
    record = verify_package(output)
    names, small = set(), {}
    # Read the entire gzip stream to check its CRC. Do not extract or load layers.
    with gzip.open(output / record['archive'], 'rb') as raw:
        with tarfile.open(fileobj=raw, mode='r|') as tar:
            for member in tar:
                name = member.name
                if Path(name).is_absolute() or '..' in Path(name).parts or not (member.isfile() or member.isdir()):
                    raise RuntimeError('Unsafe Docker archive member')
                names.add(name)
                if member.isfile() and member.size < 2 * 1024 * 1024:
                    blob = tar.extractfile(member).read()
                    try:
                        small[name] = json.loads(blob)
                    except (UnicodeDecodeError, ValueError):
                        pass
        while raw.read(4 * 1024 * 1024):
            pass
    manifests = small.get('manifest.json')
    if not manifests or len(manifests) != 1:
        raise RuntimeError('Expected one saved Docker image')
    m = manifests[0]
    if record['image'] not in m['RepoTags'] or not set(m['Layers']) <= names:
        raise RuntimeError('Missing image tag or layer in archive')
    cfg = small.get(m['Config'])
    if not cfg or cfg['rootfs']['diff_ids'] != record['rootfs_layers'] or cfg['architecture'] != 'arm64':
        raise RuntimeError('Archived image layer/platform identity differs')
    actual = hashlib.sha256(json.dumps(cfg['config'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual != record['runtime_config_sha256']:
        raise RuntimeError('Archived runtime configuration differs')
    result = {'archive_sha256': record['archive_sha256'], 'gzip_crc_verified': True,
              'all_layers_present': True, 'layers': len(m['Layers']), 'runtime_identity_verified': True,
              'docker_load_performed': False, 'production_operations': []}
    (output / 'archive-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def package_assets(output):
    source = DATA / 'offline-assets'
    record = json.loads((source / 'offline-aot.json').read_text())
    names = ['sigs.json', 'needs-rank0.json', 'needs-rank1.json', 'offline-plan.json', 'offline-aot.json'] + sorted(record['files'])
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or any((source / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)):
            raise RuntimeError('Unsafe asset archive input')
    for name, h in record['files'].items():
        if sha(source / name) != h:
            raise RuntimeError('AOT output changed: ' + name)
    output.mkdir(parents=True, exist_ok=True)
    archive, receipt = output / 'candidate-assets.tar.gz', output / 'assets.json'
    source_hash = sha(source / 'offline-aot.json')
    if archive.exists() or receipt.exists():
        old = json.loads(receipt.read_text())
        if old['source_receipt_sha256'] != source_hash or sha(archive) != old['sha256']:
            raise RuntimeError('Existing asset package differs; retain it and choose another output')
        print(json.dumps(old, indent=2))
        return
    def owner(member):
        member.uid = member.gid = 0
        member.uname = member.gname = ''
        return member
    part = output / 'candidate-assets.tar.gz.part'
    with tarfile.open(part, 'w:gz') as tar:
        for name in names:
            tar.add(source / name, arcname=name, recursive=False, filter=owner)
    part.rename(archive)
    out = {'archive': archive.name, 'sha256': sha(archive), 'bytes': archive.stat().st_size,
           'source_receipt_sha256': source_hash, 'model_bound': False, 'gpu_qualified': False}
    receipt.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


def assets(image, output, config, compile_aot=False):
    c = profile.read_config(config)
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    helper = '/opt/mainline-generators/engine/zig/tools/zig/'
    base = ['docker', 'run', '--rm', '--runtime', 'runc', '--network', 'none',
            '--cpus', '2', '--memory', '3g', '-v', str(output) + ':/out']
    receipt = output / 'offline-plan.json'
    envs = [profile.environment(c, r) for r in (0, 1)]
    identity = {'tensorfold': prepare.pins()['tensorfold']['revision'],
                'patches': [p['sha256'] for p in prepare.pins()['patches']],
                'profile_sha256': profile.digest(json.dumps(envs, sort_keys=True).encode()),
                'image_content': image_identity(json.loads(subprocess.check_output(['docker', 'image', 'inspect', image]))[0])}
    if receipt.exists() and any(json.loads(receipt.read_text())[k] != v for k, v in identity.items()):
        raise RuntimeError('Existing offline assets have another profile/image; choose a new output')
    if compile_aot:
        if not receipt.exists():
            raise RuntimeError('Run aot-plan first')
        planned = json.loads(receipt.read_text())
        for n, h in planned['inputs'].items():
            if sha(output / n) != h:
                raise RuntimeError('AOT input changed: ' + n)
        for rank in (0, 1):
            subprocess.run(base + ['--entrypoint', 'python3', image, helper + 'triton_fill.py', 'compile',
                           '--py', '/opt/mainline-python-reference/src', '--needs', '/out/needs-rank' + str(rank) + '.json',
                           '--options', helper + 'dsv41_triton_options.json', '--out', '/out/aot'], check=True)
        record = dict(identity, gpu_qualification=False, checkpoint_qualification=False,
                      files={str(p.relative_to(output)): sha(p) for p in sorted((output / 'aot').rglob('*')) if p.is_file()})
        (output / 'offline-aot.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'compiled': True, 'gpu_qualification': False, 'files': len(record['files'])}))
        return
    subprocess.run(base + ['--entrypoint', 'python3', image, helper + 'triton_fill.py', 'sigs',
                   '--py', '/opt/mainline-python-reference/src', '--out', '/out/sigs.json'], check=True)
    for rank in (0, 1):
        args = base + ['--entrypoint', 'tf-dsv41-assets']
        for k, v in sorted(envs[rank].items()):
            args += ['-e', k + '=' + v]
        subprocess.run(args + [image, 'aot-needs', '/out/sigs.json', '/out/needs-rank' + str(rank) + '.json'], check=True)
    identity['inputs'] = {n: sha(output / n) for n in ['sigs.json', 'needs-rank0.json', 'needs-rank1.json']}
    identity.update(gpu_qualification=False, checkpoint_qualification=False)
    receipt.write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(identity, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['generator', 'generator-build', 'package', 'package-assets', 'verify-package', 'inspect-archive', 'aot-plan', 'aot-compile'])
    p.add_argument('--image', default='tensorfold-dsv41:mainline-2ba55ec7-assets')
    p.add_argument('--output', type=Path)
    p.add_argument('--config', type=Path, default=ROOT / 'profile.example.json')
    a = p.parse_args()
    output = a.output or DATA / ('offline-assets' if a.operation.startswith('aot-') else 'offline-image')
    if a.operation == 'generator':
        generator()
    elif a.operation == 'generator-build':
        build_generator(a.image)
    elif a.operation.startswith('aot-'):
        assets(a.image, output, a.config, a.operation == 'aot-compile')
    elif a.operation == 'package':
        package(a.image, output)
    elif a.operation == 'package-assets':
        package_assets(output)
    elif a.operation == 'inspect-archive':
        inspect_archive(output)
    else:
        print(json.dumps(verify_package(output), indent=2))


if __name__ == '__main__':
    main()

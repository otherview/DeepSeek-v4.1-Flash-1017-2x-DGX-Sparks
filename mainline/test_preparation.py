"""Offline migration boundary checks; no model or GPU is touched."""
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import prepare
import profile
import offline


class MigrationSafety(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).parent / 'profile.example.json').read_text())

    def test_pinned_patch_inventory_and_private_data_exclusion(self):
        pins = prepare.pins()
        for p in pins['patches']:
            self.assertEqual(prepare.sha(prepare.ROOT / p['file']), p['sha256'])
        self.assertEqual(pins['tensorfold']['revision'], '2ba55ec7d32d0bec446bb012678550be7ef52895')
        self.assertIn('data/', (prepare.ROOT / '.gitignore').read_text())

    def test_archive_cannot_escape_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with tarfile.open(root / 'bad.tar', 'w') as t:
                member = tarfile.TarInfo('repo/../outside')
                member.size = 1
                t.addfile(member, io.BytesIO(b'x'))
            with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
                prepare.extract(root / 'bad.tar', root / 'source')
            self.assertFalse((root / 'outside').exists())

    def test_plan_preserves_history_limits_and_new_namespaces(self):
        c = self.config()
        rows = profile.commands(c)
        head = rows[0]['argv']
        self.assertIn('--loop-guard', head)
        for flag in ['--drop-reasoning-history', '--loop-guard-stop', '--thinking-budget']:
            self.assertNotIn(flag, head)
        self.assertEqual(head[head.index('--max-tokens') + 1], '32768')
        for rank, row in enumerate(rows):
            self.assertIn('dsv41-mainline-2ba55ec7-rank' + str(rank), row['argv'])
            self.assertNotIn('dsv41-mia-20261112-rank', row['argv'])
            self.assertEqual(profile.environment(c, rank)['TF_DSV41_PREFILL'], 'full')
            self.assertEqual(profile.environment(c, rank)['TF_DSV41_IMAGES'], '0')
        self.assertTrue(any('24-hour' in x for x in profile.activation_blockers(c)))

    def test_capacity_changes_are_refused(self):
        c = self.config()
        c['context'] = 32768
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.json'
            p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, 'context'):
                profile.read_config(p)

    def assets(self, root):
        c = self.config()
        c['context'] = 1
        c['assets'] = str(root / 'assets')
        c['checkpoint'] = str(root / 'model')
        a, model = Path(c['assets']), Path(c['checkpoint'])
        (a / 'aot').mkdir(parents=True)
        model.mkdir()
        for n in ['config.json', 'tokenizer.json']:
            (model / n).write_text('{}')
        rows = c['context'] + 2048
        (a / 'rope.bin').write_bytes(b'DSV41RP1' + struct.pack('<II', rows, 64) + b'\0' * (rows * 64 * 8))
        (a / 'engram-token-map.bin').write_bytes(b'DSENGR01')
        (a / 'aot/aot.json').write_text('{"kernels":[{"fixture":true}]}')
        pins = prepare.pins()
        receipt = {'tensorfold': pins['tensorfold']['revision'],
                   'patches': [p['sha256'] for p in pins['patches']],
                   'profile_sha256': profile.digest(json.dumps([profile.environment(c, r) for r in (0, 1)], sort_keys=True).encode()),
                   'model_files': {n: prepare.sha(model / n) for n in ['config.json', 'tokenizer.json']},
                   'files': {n: prepare.sha(a / n) for n in ['rope.bin', 'engram-token-map.bin', 'aot/aot.json']}}
        (a / 'manifest.json').write_text(json.dumps(receipt))
        return c, a, receipt

    def test_changed_assets_and_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, a, receipt = self.assets(Path(tmp))
            self.assertTrue(profile.verify_assets(c)['asset_digests_verified'])
            (a / 'engram-token-map.bin').write_bytes(b'legacy')
            with self.assertRaisesRegex(ValueError, 'digest differs'):
                profile.verify_assets(c)
            c['parallel'] = 1
            with self.assertRaisesRegex(ValueError, 'planning settings differ'):
                profile.verify_assets(c)

    def test_legacy_asset_format_is_rejected_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, a, receipt = self.assets(Path(tmp))
            (a / 'engram-token-map.bin').write_bytes(b'old-json')
            receipt['files']['engram-token-map.bin'] = prepare.sha(a / 'engram-token-map.bin')
            (a / 'manifest.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'Legacy token map'):
                profile.verify_assets(c)

    def test_package_detects_changed_or_truncated_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'candidate-image.tar.gz'
            archive.write_bytes(b'complete fixture')
            record = {'archive': archive.name, 'archive_bytes': archive.stat().st_size,
                      'archive_sha256': offline.sha(archive)}
            (root / 'image.json').write_text(json.dumps(record))
            self.assertEqual(offline.verify_package(root), record)
            archive.write_bytes(b'complete fixturf')
            with self.assertRaisesRegex(RuntimeError, 'digest/size'):
                offline.verify_package(root)
            archive.write_bytes(b'truncated')
            with self.assertRaisesRegex(RuntimeError, 'digest/size'):
                offline.verify_package(root)

    def test_cached_dependency_mismatch_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'dependency'
            p.write_bytes(b'operator data')
            with patch('urllib.request.urlretrieve') as download:
                with self.assertRaisesRegex(RuntimeError, 'Existing dependency differs'):
                    offline.fetch('https://example.invalid/dependency', p, '0' * 64)
                download.assert_not_called()
            self.assertEqual(p.read_bytes(), b'operator data')

    def test_archive_inspection_rejects_missing_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {'architecture': 'arm64', 'rootfs': {'diff_ids': ['sha256:fixture']}, 'config': {'Cmd': ['--help']}}
            manifest = [{'RepoTags': ['candidate:fixture'], 'Config': 'config.json', 'Layers': ['layer.tar']}]
            runtime = hashlib.sha256(json.dumps(config['config'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            for include_layer in [True, False]:
                archive = root / 'candidate-image.tar.gz'
                with tarfile.open(archive, 'w:gz') as tar:
                    files = {'config.json': json.dumps(config).encode(), 'manifest.json': json.dumps(manifest).encode()}
                    if include_layer:
                        files['layer.tar'] = b'fixture layer'
                    for name, blob in files.items():
                        entry = tarfile.TarInfo(name); entry.size = len(blob)
                        tar.addfile(entry, io.BytesIO(blob))
                record = {'image': 'candidate:fixture', 'archive': archive.name, 'archive_bytes': archive.stat().st_size,
                          'archive_sha256': offline.sha(archive), 'rootfs_layers': config['rootfs']['diff_ids'],
                          'runtime_config_sha256': runtime}
                (root / 'image.json').write_text(json.dumps(record))
                if include_layer:
                    offline.inspect_archive(root)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Missing image tag or layer'):
                        offline.inspect_archive(root)

    def test_image_identity_accepts_metadata_ids_but_rejects_runtime_drift(self):
        import copy
        first = {'Id': 'index-digest', 'Architecture': 'arm64',
                 'RootFS': {'Layers': ['a', 'b']}, 'Config': {'Env': ['MODE=thinking'], 'Cmd': ['--help']}}
        loaded = copy.deepcopy(first); loaded['Id'] = 'config-digest'
        self.assertEqual(offline.image_identity(first), offline.image_identity(loaded))
        changed = copy.deepcopy(loaded); changed['Config']['Env'] = ['MODE=no-thinking']
        self.assertNotEqual(offline.image_identity(first), offline.image_identity(changed))
        changed = copy.deepcopy(loaded); changed['RootFS']['Layers'].reverse()
        self.assertNotEqual(offline.image_identity(first), offline.image_identity(changed))


if __name__ == '__main__':
    unittest.main()

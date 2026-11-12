"""Verify report files and archived evidence without importing model dependencies."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

root = Path(__file__).resolve().parent
for line in (root / 'SHA256SUMS').read_text().splitlines():
    digest, relative = line.split('  ', 1)
    path = PurePosixPath(relative)
    assert not path.is_absolute() and '..' not in path.parts, relative
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
expected = json.loads((root / 'evidence-members.json').read_text())
with tarfile.open(root / 'evidence.tar.gz', 'r:gz') as archive:
    members = archive.getmembers()
    assert len(members) == len(expected) and len({m.name for m in members}) == len(members)
    for member in members:
        assert member.isfile() and member.name in expected
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == expected[member.name], member.name
diagnostics = json.loads((root / 'diagnostics.json').read_text())
assert diagnostics['split'] == 'development'
assert all(r['overall']['query_count'] == 8 for r in diagnostics['runs'])
assert all(r['overall']['coverage'] == 1 for r in diagnostics['runs'])
print(f'Verified report files and {len(expected)} archived evidence members; development fixture only')

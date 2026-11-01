"""Check a proposed clip expansion while retaining the existing source split."""
import argparse
import json
from pathlib import Path

from inventory import ROOT, sha, validate_partition


def validate_extension(rows, existing_assignments, proposals):
    if not isinstance(proposals, dict):
        raise ValueError('Proposals must map clip names to TRAIN or EVAL')
    combined = dict(existing_assignments)
    for clip, role in proposals.items():
        if clip in combined and combined[clip] != role:
            raise ValueError('Existing clip assignment cannot be silently changed')
        combined[clip] = role
    groups = validate_partition(rows, combined)
    return combined, groups


def main(inventory_path, expected_sha, proposal_path, output):
    witness = json.loads((ROOT / 'EXTENSION_TESTS.json').read_text(encoding='utf-8'))
    if witness.get('status') != 'passed' or witness.get('extension_code_sha256') != sha(ROOT / 'EXTENSION_CODE.sha256'):
        raise RuntimeError('Split-extension witness differs')
    for line in (ROOT / 'EXTENSION_CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('Split-extension source changed: ' + name)
    inventory_path = Path(inventory_path)
    if len(expected_sha) != 64 or sha(inventory_path) != expected_sha:
        raise RuntimeError('Inventory hash differs')
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    done = json.loads((ROOT / 'DONE.json').read_text(encoding='utf-8'))
    config = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    if (done.get('status') != 'complete' or done.get('inventory_sha256') != expected_sha
            or done.get('code_sha256') != sha(ROOT / 'CODE.sha256')
            or inventory.get('status') != 'complete' or inventory.get('code_sha256') != sha(ROOT / 'inventory.py')
            or inventory.get('protocol_sha256') != sha(ROOT / 'protocol.json')
            or inventory.get('existing_assignments') != config['existing_assignments']):
        raise RuntimeError('Inventory and locked existing assignments are not bound')
    proposals = {} if proposal_path is None else json.loads(Path(proposal_path).read_text(encoding='utf-8'))
    assignments, groups = validate_extension(inventory['rows'], inventory['existing_assignments'], proposals)
    by_clip = {row['clip']: row for row in inventory['rows']}
    dataset_root = Path(inventory['dataset_root']).resolve(strict=True)
    verified = []
    for clip, role in sorted(assignments.items()):
        path = (dataset_root / (clip + '.mp4')).resolve(strict=True)
        if path.parent != dataset_root or sha(path) != by_clip[clip]['source_video_sha256']:
            raise RuntimeError('Assigned raw video changed since inventory: ' + clip)
        verified.append(dict(clip=clip, role=role, source_group=by_clip[clip]['source_group'],
                             source_video_sha256=by_clip[clip]['source_video_sha256']))
    result = dict(status='passed', inventory_sha256=expected_sha, guard_code_sha256=sha(__file__),
        existing_assignments_preserved=True, proposed_assignments=proposals, verified_clips=verified,
        assigned_group_roles=groups, gpu_used=False, training_performed=False, clean_holdout_established=False,
        scope='Group- and exact-file-based split guard; original-footage duplication or previous exposure requires separate review')
    with Path(output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('PASS: locked assignments, group split and assigned raw-video hashes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--expected-inventory-sha256', required=True)
    parser.add_argument('--proposals', help='Optional JSON mapping additional clip names to TRAIN or EVAL')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.inventory, args.expected_inventory_sha256, args.proposals, args.output)

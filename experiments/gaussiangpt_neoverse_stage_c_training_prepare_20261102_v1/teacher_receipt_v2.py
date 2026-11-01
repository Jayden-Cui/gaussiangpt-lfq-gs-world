"""Require a reviewed, hash-bound, fixed decoder/free-code teacher pair."""
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def state_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        tensor=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate(path,decoder,runtime,expected_codec,expected_cache):
    """Review may be performed by the experimenter; no user approval is implied."""
    receipt=read(path)
    if (receipt.get('schema')!='continuous_action_teacher_v1' or receipt.get('status')!='reviewed'
            or receipt.get('scope')!='single_train_video_168_windows' or receipt.get('quantized') is not False
            or receipt.get('codec_sha256')!=expected_codec or receipt.get('cache_manifest_sha256')!=expected_cache):
        raise RuntimeError('Teacher receipt scope, codec, cache or review status differs')
    paths={}
    for name in ('decoder_checkpoint','codes','evaluation','native_replay','visual_review'):
        entry=receipt[name];artifact=Path(entry['path'])
        if not artifact.is_absolute() or not artifact.is_file() or sha(artifact)!=entry['sha256']:
            raise RuntimeError('Teacher artifact missing or changed: '+name)
        paths[name]=artifact
    review=read(paths['visual_review'])
    if (review.get('status')!='reviewed' or review.get('major_motion_usable') is not True
            or not {35,123}.issubset(set(review.get('reviewed_starts',[])))
            or len(set(review.get('reviewed_starts',[])))<8
            or review.get('evaluation_sha256')!=receipt['evaluation']['sha256']):
        raise RuntimeError('Teacher dynamic review is absent, failed or bound to another evaluation')
    evaluation=read(paths['evaluation'])
    source=evaluation.get('source',{})
    if (evaluation.get('status')!='complete' or evaluation.get('window_predictions')!=1512
            or evaluation.get('unique_future_frames')!=176
            or evaluation.get('all_source_frames_preserved') is not True
            or evaluation.get('decoder_future_geometry_input') is not False
            or source.get('decoder_digest')!=receipt['decoder_tensor_digest']
            or source.get('codec_sha256')!=expected_codec or source.get('cache_manifest_sha256')!=expected_cache
            or source.get('merged_codes_sha256',source.get('codes_sha256'))!=receipt['codes']['sha256']):
        raise RuntimeError('Teacher evaluation coverage or decoder differs')
    future=[row for row in evaluation.get('rows',[]) if row.get('horizon',0)>0]
    pairs={(row['start'],row['horizon']) for row in future}
    if (len(future)!=1512 or pairs!={(s,h) for s in range(168) for h in range(1,10)}
            or any('correct' not in row.get('metrics',{}) or 'correct' not in row.get('support',{}) for row in future)):
        raise RuntimeError('Teacher evaluation rows are incomplete, duplicated or lack predictions')
    replay=read(paths['native_replay'])
    packet_sha=evaluation.get('packets_sha256') or evaluation.get('raw_original_packets_sha256')
    if (replay.get('status')!='passed' or replay.get('packets')!=104 or replay.get('views')!=208
            or replay.get('packets_sha256')!=packet_sha or replay.get('encoder_or_codec_loaded') is not False):
        raise RuntimeError('Independent native GS replay is absent or bound to other packets')
    recorded=replay.get('rows',[]);replay_rows={row['name']:row for row in recorded}
    required={f's{s:03d}_h{h:02d}_correct' for s in (0,35,56,84,112,123,140,167) for h in range(10)}
    if (len(recorded)!=104 or len(replay_rows)!=104 or not required.issubset(replay_rows)
            or any(not replay_rows[name].get('passed') for name in required)):
        raise RuntimeError('Correct-code native packets did not all pass replay')
    if runtime.digest(decoder)!=receipt['decoder_tensor_digest'] or any(x.requires_grad for x in decoder.parameters()):
        raise RuntimeError('Decoder differs from reviewed teacher or is not frozen')
    checkpoint=torch.load(paths['decoder_checkpoint'],map_location='cpu',weights_only=True)
    state=checkpoint[receipt['decoder_checkpoint'].get('state_key','decoder')]
    if state_digest(state)!=receipt['decoder_tensor_digest']:
        raise RuntimeError('Checkpoint bytes do not contain the reviewed frozen decoder')
    payload=torch.load(paths['codes'],map_location='cpu',weights_only=True)
    if (payload.get('starts')!=list(range(168)) or payload.get('decoder_digest')!=receipt['decoder_tensor_digest']):
        raise RuntimeError('Free-code ordering or paired decoder differs')
    codes=payload['codes']
    if (codes.dtype!=torch.float32 or tuple(codes.shape)!=(168,9,16)
            or not bool(torch.isfinite(codes).all())):
        raise RuntimeError('Teacher must contain finite continuous [168,9,16] codes')
    return dict(receipt=receipt,receipt_sha256=sha(path),codes=codes,
                decoder_checkpoint=paths['decoder_checkpoint'],evaluation=evaluation)

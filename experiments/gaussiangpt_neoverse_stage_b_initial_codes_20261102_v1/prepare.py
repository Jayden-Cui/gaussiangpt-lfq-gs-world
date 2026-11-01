"""CPU-only initialization in the successful two-window decoder's code gauge."""
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT=Path(__file__).resolve().parent
B_ROOT=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1')
sys.path.insert(0,str(B_ROOT))
from b_core import initialize_codes,load_bound_sources


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value,replace=False):
    path=Path(path);text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(path)
    else:
        with path.open('x',encoding='utf-8') as stream:stream.write(text)


def assemble(chunks):
    values=torch.empty(168,9,16,dtype=torch.float32);seen=set()
    for starts,codes in chunks:
        if tuple(codes.shape)!=(len(starts),9,16) or codes.dtype!=torch.float32:
            raise ValueError('Invalid initialization chunk')
        for i,start in enumerate(starts):
            if start in seen or not 0<=start<168:raise ValueError('Duplicate/invalid start')
            values[start]=codes[i];seen.add(start)
    if seen!=set(range(168)) or not bool(torch.isfinite(values).all()):
        raise ValueError('Incomplete/nonfinite initialization')
    return values


def main():
    began=time.monotonic();phase='initializing'
    write(ROOT/'STARTED.json',dict(pid=__import__('os').getpid(),device='cpu'))
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        test=json.loads((ROOT/'CPU_TESTS.json').read_text())
        if test['status']!='passed' or test['code_sha256']!=sha(ROOT/'CODE.sha256'):
            raise RuntimeError('Initialization orchestration witness changed')
        for line in (ROOT/'CODE.sha256').read_text().splitlines():
            digest,name=line.split('  ',1)
            if Path(name).name!=name or sha(ROOT/name)!=digest:raise RuntimeError('Initialization source changed: '+name)
        prior=json.loads((B_ROOT/'CPU_TESTS.json').read_text())
        if prior['status']!='passed' or prior['code_sha256']!=sha(B_ROOT/'CODE.sha256'):
            raise RuntimeError('Stage-B model witness changed')
        for line in (B_ROOT/'CODE.sha256').read_text().splitlines():
            digest,name=line.split('  ',1)
            if Path(name).name!=name or sha(B_ROOT/name)!=digest:raise RuntimeError('Stage-B source changed: '+name)
        bundle=load_bound_sources('cpu');p=bundle['p'];decoder=bundle['decoder'].eval();posterior=bundle['posterior'].eval()
        runtime=bundle['runtime'];decoder_before=runtime.digest(decoder);posterior_before=runtime.digest(posterior)
        chunks=[];phase='extracting_codes'
        for offset in range(0,168,8):
            if time.monotonic()-began>1800:raise RuntimeError('CPU initialization deadline')
            starts=list(range(offset,min(offset+8,168)))
            codes=initialize_codes(posterior,decoder,bundle['store'],starts,'cpu')
            chunks.append((starts,codes))
            write(ROOT/'STATUS.json',dict(status='running',phase=phase,completed_windows=offset+len(starts),
                  total_windows=168,elapsed_seconds=time.monotonic()-began),replace=True)
        result=assemble(chunks)
        if not torch.equal(result[35],decoder.codes[0].detach()) or not torch.equal(result[123],decoder.codes[1].detach()):
            raise RuntimeError('Original two-window free codes not preserved')
        if runtime.digest(decoder)!=decoder_before or runtime.digest(posterior)!=posterior_before:
            raise RuntimeError('Frozen model changed during CPU initialization')
        source=dict(decoder_digest=decoder_before,posterior_digest=posterior_before,
          decoder_checkpoint_sha256=p['teacher_checkpoint_sha256'],posterior_checkpoint_sha256=p['posterior_checkpoint_sha256'],
          codec_sha256=p['codec_checkpoint_sha256'],cache_manifest_sha256=sha(bundle['paths']['cache_manifest']),
          preparation_code_sha256=sha(ROOT/'CODE.sha256'),stage_b_core_sha256=sha(B_ROOT/'CODE.sha256'),
          anchors=[35,123],quantized=False)
        path=ROOT/'INITIAL_CODES.pt'
        with path.open('xb') as stream:
            torch.save(dict(schema='stage_b_initial_continuous_codes_v1',starts=list(range(168)),
                            codes=result,decoder_digest=decoder_before,source=source),stream)
        replay=torch.load(path,map_location='cpu',weights_only=True)
        if replay['starts']!=list(range(168)) or not torch.equal(replay['codes'],result):
            raise RuntimeError('Saved initial codes did not round-trip')
        write(ROOT/'INITIAL_CODES.json',dict(status='complete',shape=[168,9,16],windows=168,
              file=path.name,sha256=sha(path),source=source,training_performed=False,
              anchor_codes_preserved=True,decoder_and_qformer_unchanged=True,elapsed_seconds=time.monotonic()-began,
              scope='CPU initialization only; not trained free codes or an evaluated teacher'))
        write(ROOT/'STATUS.json',dict(status='complete',phase='initialization',completed_windows=168,
              total_windows=168,training_performed=False,elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as error:
        failure=dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(ROOT/'FAILED.json',failure);write(ROOT/'STATUS.json',failure,replace=True)
        raise


if __name__=='__main__':main()

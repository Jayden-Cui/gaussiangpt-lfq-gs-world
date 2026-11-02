"""Read-only CPU audit of green-beam data actually used by the round-five run."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

BASE = Path('/data/CJH/AGS-V6Direct/outputs')
FULL = BASE/'gaussiangpt_neoverse_qformer_full177_20261101_v1'
RUN = BASE/'gaussiangpt_neoverse_anchor_replay_pilot_20261102_v1'
NATIVE = BASE/'neoverse_tower177_fps30_20261029_v1/native448'
STATE = BASE/'gaussiangpt_neoverse_state_motion_20261030_v1/state'
FORMAL = BASE/'gaussiangpt_neoverse_z_paired_formal_20261031_v1'
OUT = Path(__file__).resolve().parent/'result'
GREEN_FRAMES = list(range(112, 150))
GREEN_STARTS = list(range(112, 141))


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def checked(p, digest):
    check(sha(p) == digest, 'Source changed: '+str(p))


def as_u8(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    return np.rint(np.clip(x, 0, 1)*255).astype(np.uint8)


def video(path, frames, fps=5):
    h, w = frames[0].shape[:2]
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
           '-an', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(path)]
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True, timeout=60)
    return dict(file=path.name, sha256=sha(path), frames=len(frames), playback_fps=fps,
                interpolation=False, header_labels=False)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    began = time.monotonic()
    check(not OUT.exists(), 'Output already exists; do not overwrite evidence')
    OUT.mkdir()
    p = read(FULL/'protocol.json')
    for name in ('codec_binding.py', 'teacher_binding.py'):
        checked(FORMAL/name, p['dependencies'][name])
    sys.path.insert(0, str(FORMAL))
    from codec_binding import BaselineStates
    from teacher_binding import BoundTeachers
    spec = importlib.util.spec_from_file_location('green_proxy', FULL/'core.py')
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    for line in (FULL/'CODE.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if name in ('core.py', 'data.py', 'prepare.py'):
            checked(FULL/name, digest)

    done = read(RUN/'run/training/TRAIN_DONE.json')
    summary = read(RUN/'run/SUMMARY.json')
    checked(RUN/'run/training/TRAIN_DONE.json', summary['train_done_sha256'])
    checked(FULL/'cache/MANIFEST.json', done['source']['cache_manifest_sha256'])
    manifest = read(FULL/'cache/MANIFEST.json')
    checked(FULL/'protocol.json', manifest['protocol_sha256'])
    check(manifest['starts'] == list(range(168)), 'Wrong complete-window sequence')
    for start in range(168):
        check(manifest['windows'][str(start)]['target_frames'] == list(range(start+1, start+10)),
              'Window frame mapping changed')
    entry = manifest['frames']
    checked(FULL/'cache'/entry['file'], entry['sha256'])
    frames = torch.load(FULL/'cache'/entry['file'], map_location='cpu', weights_only=True)
    states = BaselineStates(BASE/'gaussiangpt_neoverse_codec_complete_eval_20261031_v1/evaluations/baseline')
    teachers = BoundTeachers(states)
    source_states = [states.load(f) for f in range(177)]
    check(frames['state_sources'] == [s['state_sha256'] for s in source_states], 'Cached state identities differ')
    check(len(frames['samples']) == 177, 'Frame cache count differs')
    print('Loaded and verified actual LDM frame cache and all 177 state packets', flush=True)

    mapping_path = STATE/'mappings/buildingblock_tower_17_176.json'
    mapping = read(mapping_path)
    prov = read(NATIVE/'provenance.json')
    source_video = Path(prov['input']['path'])
    checked(source_video, prov['input']['source_video_sha256'])
    checked(NATIVE/'input_rgb.npy', teachers.preparation['rgb_sha256'])
    rgb = np.load(NATIVE/'input_rgb.npy', mmap_mode='r')
    native_images = np.load(NATIVE/'rendered_rgb.npy', mmap_mode='r')
    cameras = np.load(NATIVE/'cameras.npz')
    check(rgb.shape == native_images.shape == (177, 252, 448, 3), 'Native image shapes differ')
    check(hashlib.sha256(rgb.tobytes()).hexdigest() == prov['input']['processed_rgb_sha256'], 'Processed RGB fingerprint differs')
    check(mapping['source_frame_indices'] == mapping['model_timestamps'] == list(range(177)), 'Frame sequence gap')
    check(mapping['source_sha256'] == prov['input']['source_video_sha256'], 'Video identity differs')
    pts = [s['pts'] for s in source_states]
    check(pts == mapping['source_pts_seconds'] == prov['input']['source_timestamps_seconds'], 'State/source timestamps differ')
    check(np.allclose(np.diff(pts), 1/29.97, atol=1e-12, rtol=0), 'Unexpected frame interval')
    check(np.array_equal(cameras['timestamps'], np.arange(177)), 'Native camera timestamp mismatch')
    for f, sample in enumerate(frames['samples']):
        expected = torch.from_numpy(np.array(rgb[f])).float()/255
        check(torch.equal(torch.from_numpy(as_u8(sample['record']['real'])),torch.from_numpy(np.array(rgb[f])))
              and float((sample['record']['real']-expected).abs().max())<=1e-7,
              f'LDM RGB differs from native input at {f}')

    # Use installed system FFmpeg: the codec environment lacks imageio-ffmpeg.
    # Preserve v1 failure; do not install packages or change the preprocessing.
    streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration',
        '-of','json',str(source_video)],text=True))['streams']
    source_meta=streams[0]
    num,den=map(int,source_meta['avg_frame_rate'].split('/'));source_meta['fps']=num/den
    width,height=source_meta['width'],source_meta['height'];frame_bytes=width*height*3
    command=['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',str(source_video),
             '-map','0:v:0','-frames:v','177','-pix_fmt','rgb24','-vsync','0','-threads','1','-f','rawvideo','pipe:1']
    redecode=[]
    with (OUT/'ffmpeg_source_decode.log').open('xb') as errors:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=errors)
        try:
            for f in range(177):
                raw=process.stdout.read(frame_bytes)
                check(len(raw)==frame_bytes,'Original video decode ended early at '+str(f))
                image=np.frombuffer(raw,dtype=np.uint8).reshape(height,width,3)
                transform=prov['input']['preprocessing']['transforms'][f]
                im=Image.fromarray(image).convert('RGB')
                check([im.height,im.width]==transform['original_hw'],'Original dimensions differ')
                rh,rw=transform['resized_hw']
                processed=np.asarray(im.resize((rw,rh),Image.Resampling.LANCZOS).crop(transform['crop_xyxy']))
                redecode.append(dict(frame=f,exact=bool(np.array_equal(processed,rgb[f])),
                    pixel_mae=float(np.abs(processed.astype(float)-rgb[f].astype(float)).mean()),
                    best_local_match=min(range(max(0,f-3),min(177,f+4)),
                        key=lambda q:float(np.abs(processed.astype(float)-rgb[q].astype(float)).mean()))))
            check(process.wait(timeout=60)==0,'Source video decoder failed')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate();process.wait(timeout=15)
    check(len(redecode)==177,'Original MP4 has fewer than 177 frames')
    print('Source MP4 decode compared with all 177 training RGB frames', flush=True)

    raw_rows = []
    for f in GREEN_FRAMES:
        raw = teachers.load_raw(f)
        cached = frames['samples'][f]
        for key in ('real','cameras','intrinsics'):
            check(torch.equal(raw[key],cached['record'][key]), f'Cached {key} differs at {f}')
        check(torch.equal(raw['targets'],cached['teacher_targets']), f'Teacher target cache differs {f}')
        check(torch.equal(teachers.masks[f],cached['mask']), f'Teacher valid mask differs {f}')
        for key in ('near','far'):
            check(raw[key] == cached['record'][key], f'Camera clipping differs {f}')
        native = teachers.native_root/f'native_group_{f:03d}.npz'
        checked(native, teachers.assets['records'][f]['source_sha256'])
        with np.load(native) as a:
            meta = json.loads(str(a['metadata_json']))
            check(meta['timestamp']==f, f'Native GS timestamp differs {f}')
            check(meta['forward_vel'] is None and meta['backward_vel'] is None, 'Unexpected interpolated motion')
            for key in ('means','harmonics','scales','rotations','opacities'):
                check(a[key].shape[0]==112896 and np.isfinite(a[key]).all(), f'Invalid GS {key}, frame {f}')
            check((a['scales']>0).all(), 'Non-positive scale')
            check(((a['opacities']>=0)&(a['opacities']<=1)).all(), 'Invalid opacity')
        render_delta=float(np.abs(raw['targets'][0].numpy()-native_images[f]).mean())
        check(render_delta<=1e-5, f'Native teacher image mismatch {f}')
        camera=cameras['camera_to_world'][f].copy();camera[:3,3]*=raw['normalization']['scale']
        check(np.allclose(camera,raw['cameras'][0].numpy(),atol=1e-7,rtol=0), f'Camera/source mismatch {f}')
        raw_rows.append(dict(frame=f, pts=pts[f], native_gs=112896, latent_nodes=len(source_states[f]['coordinates']),
            native_to_cached_teacher_mae=render_delta, native_sha256=sha(native),
            state_sha256=source_states[f]['state_sha256'], raw_cache_sha256=teachers.assets['records'][f]['cache_sha256']))

    window_rows=[]
    for start in GREEN_STARTS:
        entry=manifest['windows'][str(start)];path=FULL/'cache'/entry['file'];checked(path,entry['sha256'])
        w=torch.load(path,map_location='cpu',weights_only=True)
        check(w['start']==start and len(w['states'])==10, 'Window identity mismatch')
        c=w['condition'];t=w['targets'];check(len(torch.unique(c['coordinates'],dim=0))==len(c['coordinates']), 'Duplicate candidates')
        for h in range(10):
            actual=w['states'][h];expected=source_states[start+h]
            for key in ('coordinates','features'):
                check(torch.equal(actual[key],expected[key]),f'Window {start} h{h} {key} misaligned')
            for key in ('frame_index','pts','clip','state_sha256','stride','grid'):
                check(actual[key]==expected[key],f'Window state metadata differs {start}/{h}/{key}')
            if h:
                index=t['indices'][h-1]
                check(torch.equal(c['coordinates'][index],actual['coordinates']),f'Target coordinates misaligned {start}/{h}')
                check(torch.equal(t['features'][h-1],actual['features']),f'Target features misaligned {start}/{h}')
                labels=torch.zeros(len(c['coordinates']));labels[index]=1
                check(torch.equal(t['occupancy'][h-1],labels),f'Occupancy target misaligned {start}/{h}')
        for key in ('coordinates','features'):
            check(torch.equal(c['initial'][key],source_states[start][key]), 'Decoder initial state mismatch')
        window_rows.append(dict(start=start,sha256=entry['sha256'],states=10,all_target_cells_covered=True,
                                target_features_exact=True,candidates=len(c['coordinates'])))
    print('Verified 38 green-action raw frames and 29 overlapping window payloads', flush=True)

    spec=importlib.util.spec_from_file_location('pilot_sampling',RUN/'sampling.py')
    sampler=importlib.util.module_from_spec(spec);spec.loader.exec_module(sampler)
    for line in (RUN/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if name=='sampling.py':checked(RUN/name,digest)
    protocol=read(RUN/'protocol.json')
    schedule=sampler.schedule(protocol['sweeps'],protocol['base_sampler_seed'],protocol['replay_sampler_seed'])
    log=RUN/'run/training/train.jsonl'
    logged=[json.loads(x) for x in log.read_text().splitlines()]
    check(len(logged)==len(schedule)==done['updates']==2115,'Unexpected training log length')
    counts=Counter();kinds={name:Counter() for name in ('base','replay')};physical=Counter()
    for expected,actual in zip(schedule,logged):
        check(all(expected[k]==actual[k] for k in expected),'Executed sampling differs from plan')
        check(len(actual['windows'])==4,'Loss entries missing')
        for member,result in zip(actual['batch'],actual['windows']):
            check(all(member[k]==result[k] for k in member),'Rendered loss came from another sample')
            check(all(math.isfinite(result[k]) for k in ('loss','rgb_loss','latent_loss','occupancy_loss','feature_loss')),'Nonfinite sample loss')
            key=(member['start'],member['horizon']);counts[key]+=1;kinds[actual['kind']][key]+=1;physical[sum(key)]+=1
    check(len(counts)==1512 and set(kinds['base'].values())=={5},'Full-pair coverage missing')
    check(set(kinds['replay'])=={(s,h) for s in (35,123) for h in range(1,10)} and set(kinds['replay'].values())=={50},'Replay count mismatch')

    evaluation=RUN/'run/training/sweep_05/free_z/evaluation'
    checked(evaluation/'EVALUATION.json',summary['evaluation_sha256'])
    ev=read(evaluation/'EVALUATION.json');lookup={(r['start'],r['horizon']):r for r in ev['rows']}
    hand=np.zeros((252,448),bool)
    for x0,y0,x1,y1 in p['hand_roi_xyxy']:hand[y0:y1,x0:x1]=True
    spec=p['object_proxies']['123'];visual=[];video_frames=[];fourway=[]
    for h in range(10):
        f=123+h;target=rgb[f]
        path=evaluation/'snapshots/s123'/f'frame_{h:02d}.png'
        panel=np.asarray(Image.open(path).convert('RGB'));check(panel.shape==(276,1792,3),'Snapshot dimensions')
        check(np.array_equal(panel[24:276,:448],target),'Snapshot target is not the source frame')
        predicted=panel[24:276,448:896];codec=panel[24:276,1344:1792];teacher=as_u8(frames['samples'][f]['teacher_targets'][0])
        mask=proxy.color_mask(target,spec);n=int(mask.sum())
        motion=np.clip(np.abs(target.astype(np.float32)/255-rgb[123].astype(np.float32)/255).mean(-1)/.1,0,1)
        weights=np.ones((252,448))/(252*448)+hand.astype(float)/hand.sum()+(10/9)*motion/max(float(motion.sum()),1e-6)
        metrics={name:proxy.object_proxy(im,target,spec) for name,im in (('native_teacher',teacher),('actual_state_codec',codec),('ldm',predicted))}
        visual.append(dict(frame=f,horizon=h,green_pixels=n,green_image_fraction=n/(252*448),
            green_inside_hand_roi_fraction=float((mask&hand).sum()/max(n,1)),
            green_rgb_loss_coefficient_fraction=float(weights[mask].sum()/weights.sum()),
            proxy=metrics,snapshot_sha256=sha(path),
            float_motion_mae={name:lookup[(123,h)]['metrics'][key]['metrics']['motion_mae'] for name,key in
                [('ldm','correct'),('actual_state_codec','actual_target')]} if h else {'ldm':0.0}))
        video_frames.append(np.concatenate([target,codec],1))
        fourway.append(np.concatenate([target,teacher,codec,predicted],1))
    media=[video(OUT/'green_source_vs_state_codec_5fps.mp4',video_frames),
           video(OUT/'green_source_neoverse_codec_ldm_5fps.mp4',fourway)]
    raw_video=[np.concatenate([rgb[f],as_u8(frames['samples'][f]['teacher_targets'][0])],1) for f in GREEN_FRAMES]
    media.append(video(OUT/'green_context_source_vs_neoverse_5fps.mp4',raw_video))
    for h in (1,3,6,9):Image.fromarray(fourway[h]).save(OUT/f'green_h{h:02d}_fourway.png')
    contact=Image.fromarray(np.concatenate([fourway[h] for h in (1,3,6,9)],0));contact.save(OUT/'green_fourway_contact.png')
    future=visual[1:]
    report=dict(status='complete',scope='CPU dataset inspection, no model updates or fresh GPU rendering',
        clip=p['clip'],metadata_frames=177,actual_state_packets_checked=177,raw_green_frames_checked=GREEN_FRAMES,
        actual_window_payloads_checked=GREEN_STARTS,source_video=str(source_video),source_video_sha256=sha(source_video),
        source_fps=prov['input']['source_fps'],frame_gap=1,source_duration_last_pts=pts[-1],
        selected_window=dict(start=123,initial_pts=pts[123],last_pts=pts[132],frames=list(range(123,133))),
        source_redecode=dict(exact_frames=sum(x['exact'] for x in redecode),count=len(redecode),
            max_pixel_mae=max(x['pixel_mae'] for x in redecode),local_best_matches=sum(x['best_local_match']==x['frame'] for x in redecode),
            decoder_fps=source_meta.get('fps'),rows=redecode),
        full_sequence_frame_mapping_exact=True,all_training_rgb_matches_native_input=True,
        cache_manifest_sha256=sha(FULL/'cache/MANIFEST.json'),frames_cache_sha256=manifest['frames']['sha256'],
        source_mapping_sha256=sha(mapping_path),train_log_sha256=sha(log),
        all_training_samples=4*len(logged),actual_updates=len(logged),unique_window_horizon_pairs=len(counts),
        green_window_samples={str(h):dict(base=kinds['base'][(123,h)],replay=kinds['replay'][(123,h)],total=counts[(123,h)]) for h in range(1,10)},
        green_physical_frame_samples={str(f):physical[f] for f in range(124,133)},
        raw_checks=raw_rows,window_checks=window_rows,visual_rows=visual,media=media,
        green_summary=dict(native_teacher_detected=sum(x['proxy']['native_teacher']['prediction']['valid'] for x in future),
            codec_detected=sum(x['proxy']['actual_state_codec']['prediction']['valid'] for x in future),
            ldm_detected=sum(x['proxy']['ldm']['prediction']['valid'] for x in future),future_frames=9,
            mean_codec_color_iou=float(np.mean([x['proxy']['actual_state_codec']['iou'] for x in future])),
            mean_ldm_color_iou=float(np.mean([x['proxy']['ldm']['iou'] for x in future])),
            codec_motion_mae=float(np.mean([x['float_motion_mae']['actual_state_codec'] for x in future])),
            ldm_motion_mae=float(np.mean([x['float_motion_mae']['ldm'] for x in future])),
            mean_green_image_fraction=float(np.mean([x['green_image_fraction'] for x in future])),
            mean_green_inside_hand_roi_fraction=float(np.mean([x['green_inside_hand_roi_fraction'] for x in future])),
            mean_green_rgb_coefficient_fraction=float(np.mean([x['green_rgb_loss_coefficient_fraction'] for x in future]))),
        limitations=['HSV/ROI is an auxiliary proxy, not object identity or 3D ground truth.',
            'Codec pictures are stored fifth-round evaluation renders; this inspection does not run a fresh codec render.',
            'No cross-video diversity or causal online GS extraction claim; NeoVerse teacher used the full source clip.',
            'RGB coefficient fractions are not measured gradient contributions or causal explanations.'],
        script_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-began)
    (OUT/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','source_fps','selected_window','actual_updates','all_training_samples','green_window_samples','green_physical_frame_samples','green_summary','elapsed_seconds')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        import traceback
        if OUT.exists():
            (OUT/'FAILED.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2))
        raise

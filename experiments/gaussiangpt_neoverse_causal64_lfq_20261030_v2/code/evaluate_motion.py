"""All-pair motion hard-ID and continuous state replay, held-state and shuffled-motion controls."""
import argparse
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from motion_common import (BASE, CLIPS, H, W, read, write, sha, frozen_codec, load_state,
    load_pair, load_target, render_points, metrics, region_metrics, ssim, pair_ids)
from qmotion import build_model, select_support
from contracts import verify_sources
from causal_data import load_prefix


def image_metrics(image, target, boxes):
    return {**metrics(image,target), **region_metrics(image,target,boxes), 'ssim': ssim(image,target)}


@torch.no_grad()
def decode_prediction(codec, prediction, max_tokens):
    coordinates, features, keep = select_support(prediction, max_tokens=max_tokens)
    points = codec.decode_state(coordinates, features, 4)
    return points, coordinates, features, keep



def mean_metrics(rows, key):
    return {name: float(np.mean([r[key][name] for r in rows])) for name in rows[0][key]}


def make_panel(images, start, end, dt):
    labels = ['Real target', 'Hold initial state', 'Frozen target codec', 'Causal-prefix Z', 'Shuffled motion Z', 'Two-frame-context Z']
    panel = Image.new('RGB',(W*6,H+32),'#16202c')
    draw = ImageDraw.Draw(panel)
    for i,(label,image) in enumerate(zip(labels,images)):
        array = (image.detach().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
        panel.paste(Image.fromarray(array),(i*W,32))
        draw.text((i*W+6,3),label,fill='white')
        draw.text((i*W+6,17),f'{start:03d} -> {end:03d} | dt={dt:.4f}s | oracle TRAIN',fill='#bbcddd')
    return panel


class Video:
    def __init__(self,path,fps):
        self.error = path.with_suffix('.ffmpeg.log').open('wb')
        self.process = subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W*6}x{H+32}',
            '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=self.error)
    def write(self,image):
        self.process.stdin.write(np.asarray(image).tobytes())
    def close(self):
        self.process.stdin.close()
        result = self.process.wait(timeout=60)
        self.error.close()
        if result: raise RuntimeError(f'ffmpeg failed with exit code {result}')


@torch.no_grad()
def evaluate(root,arm):
    verify_sources(root)
    protocol = read(root/'protocol.json'); config = next(a for a in protocol['arms'] if a['name']==arm)
    torch.set_num_threads(6); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.60)
    folder=root/arm; output=folder/'evaluation';output.mkdir(parents=True,exist_ok=True)
    training=read(folder/'train_done.json'); checkpoint=folder/'motion_final.pt'
    assert sha(checkpoint)==training['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['protocol_sha256']==sha(root/'protocol.json')
    model=build_model(config).cuda().eval();model.load_state_dict(saved['model'])
    del saved
    codec=frozen_codec(root);regions=read(BASE/'protocol.json')['evaluation']['region_boxes']
    binding={'motion_checkpoint_sha256':sha(checkpoint),'state_checkpoint_sha256':protocol['source_checkpoint_sha256'],
        'protocol_sha256':sha(root/'protocol.json'),'source_binding_sha256':sha(root/'source_binding.json')}
    pairs=pair_ids(protocol);zs={};started=time.monotonic()
    for clip,start,end in pairs:
        window=load_prefix(root,clip,end,protocol['causal_temporal']['max_context_frames'])
        encoded=model.encode_prefix(window)
        short=model.encode_prefix(window[-2:]) if len(window)>2 else encoded
        zs[(clip,start)]={k:encoded[k].detach().cpu() for k in ['z_prequant','z_quantized','indices']}
        zs[(clip,start)]['two_frame_indices']=short['indices'].detach().cpu()
        zs[(clip,start)]['context_frames']=len(window)
        zs[(clip,start)]['history_start']=end-len(window)+1
        del window,encoded,short
    # Diagnostics contain continuous Z; inference packets below contain hard IDs only.
    torch.save({'schema':'motion-z-diagnostic-v1','binding':binding,
        'pairs':[{'clip':c,'start':s,'end':e,**zs[(c,s)]} for c,s,e in pairs]},output/'motion_z.pt')
    rows=[];manifest=[];replay_report=None
    per_clip=177-protocol['frame_gap']
    for clip in CLIPS:
        clipdir=output/clip;clipdir.mkdir(parents=True,exist_ok=True)
        mapping=read((BASE/'mappings')/f'{clip}.json')
        fps=1/float(np.median(np.diff(mapping['source_pts_seconds'])))
        video=Video(clipdir/'comparison.mp4',fps)
        try:
            for start in range(per_clip):
                end=start+protocol['frame_gap']
                s,t,initial,condition,target_indices=load_pair(root,clip,start,end,codec)
                target=load_target(clip,end)
                labels=torch.load(root/'labels'/clip/f'{start:03d}.pt',weights_only=True,map_location='cpu')
                ids=zs[(clip,start)]['indices'].cuda().long()
                prediction=model.decode_ids(ids,condition)
                points,coords,state_features,keep=decode_prediction(codec,prediction,protocol['support']['max_predicted_tokens'])
                correct=render_points(points,target,0);nearby=render_points(points,target,1)
                packet=model.packet(initial,ids,binding=binding,
                    metadata={'clip':clip,'start':start,'end':end,'normalization':s['normalization']})
                packet_path=clipdir/f'{start:03d}_motion.pt';torch.save(packet,packet_path)
                future_path=clipdir/f'{start:03d}_predicted_state.pt'
                torch.save({'schema':'gsworld_continuous_state_v1','coordinates':coords.cpu(),'features':state_features.cpu().float(),
                    'stride':4,'grid':.008,'metadata':packet['metadata'],'binding':binding},future_path)
                if replay_report is None:
                    replay=model.replay(torch.load(packet_path,weights_only=True),codec,binding=binding)
                    feature_error=float((replay['features']-prediction['features']).abs().max())
                    occupancy_error=float((replay['occupancy_logits']-prediction['occupancy_logits']).abs().max())
                    rpoints,rcoords,rfeatures,rkeep=decode_prediction(codec,replay,protocol['support']['max_predicted_tokens'])
                    state_exact=torch.equal(rcoords,coords) and torch.allclose(rfeatures,state_features,atol=1e-5,rtol=0)
                    rimage=render_points(rpoints,target)
                    render_check=metrics(rimage,correct)
                    assert feature_error<1e-5 and occupancy_error<1e-5 and state_exact
                    assert render_check['psnr']>=60 and render_check['mae']<=1e-5
                    replay_report={'passed':True,'feature_max_abs':feature_error,'occupancy_max_abs':occupancy_error,
                        'state_coordinates_and_features_exact':state_exact,'render':render_check,
                        'render_max_abs':float((rimage-correct).abs().max())}
                    del replay,rpoints,rcoords,rfeatures,rkeep,rimage
                wrong_start=(start+85)%per_clip
                wrong_ids=zs[(clip,wrong_start)]['indices'].cuda().long()
                wrong_prediction=model.decode_ids(wrong_ids,condition)
                wrong_points,_,_,wrong_keep=decode_prediction(codec,wrong_prediction,protocol['support']['max_predicted_tokens'])
                wrong=render_points(wrong_points,target)
                short_ids=zs[(clip,start)]['two_frame_indices'].cuda().long()
                short_prediction=model.decode_ids(short_ids,condition)
                short_points,_,_,_=decode_prediction(codec,short_prediction,protocol['support']['max_predicted_tokens'])
                short_image=render_points(short_points,target)
                short_metrics=image_metrics(short_image,target['real'],regions[clip])
                label_mask=torch.zeros_like(keep);label_mask[target_indices]=True
                tp=int((keep&label_mask).sum())
                correct_metrics=image_metrics(correct,target['real'],regions[clip])
                wrong_metrics=image_metrics(wrong,target['real'],regions[clip])
                row={'clip':clip,'start':start,'end':end,'dt':initial.dt,
                    'held_initial':labels['held_initial_real'],'frozen_target':t['baseline_real'],
                    'correct_motion':correct_metrics,'shuffled_motion':wrong_metrics,'two_frame_context':short_metrics,
                    'two_frame_motion_ids':short_ids.cpu().tolist()[0],
                    'history_changes_ids':not torch.equal(ids,short_ids),
                    'context_frames':zs[(clip,start)]['context_frames'],'history_start':zs[(clip,start)]['history_start'],
                    'correct_vs_frozen_target':metrics(correct,t['reconstruction'][0]),
                    'pseudo_view_teacher':metrics(nearby,target['targets'][1]),
                    'motion_ids':ids.cpu().tolist()[0],'shuffled_source_start':wrong_start,
                    'shuffle_id_changed':not torch.equal(ids,wrong_ids),
                    'predicted_tokens':len(coords),'shuffled_predicted_tokens':int(wrong_keep.sum()),
                    'occupancy_precision':tp/len(coords),'occupancy_recall':tp/len(target_indices),
                    'candidate_count':len(keep),'serialized_packet_bytes':packet_path.stat().st_size,
                    'ideal_motion_bits':config['motion_tokens']*math.log2(model.vocabulary)}
                rows.append(row)
                with (output/'pairs.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
                manifest.append({'clip':clip,'start':start,'motion_packet_sha256':sha(packet_path),'predicted_state_sha256':sha(future_path)})
                panel=make_panel([target['real'],labels['held_initial_image'],t['reconstruction'][0],correct,wrong,short_image],start,end,initial.dt)
                video.write(panel)
                if start%30==0:
                    panel.save(clipdir/f'comparison_{start:03d}.png')
                    write(folder/'eval_progress.json',{'pairs':len(rows),'total':len(pairs),'elapsed_seconds':time.monotonic()-started})
                    print('EVAL',arm,clip,start,'pairs',len(rows),flush=True)
                del s,t,initial,condition,target_indices,target,prediction,points,correct,nearby,wrong_prediction,wrong_points,wrong,keep,wrong_keep,short_prediction,short_points,short_image
        finally:
            video.close()
    controls=['held_initial','frozen_target','correct_motion','shuffled_motion','two_frame_context']
    summary={key:mean_metrics(rows,key) for key in controls}
    summary['per_clip']={clip:{key:mean_metrics([r for r in rows if r['clip']==clip],key) for key in controls} for clip in CLIPS}
    tuples=Counter(tuple(r['motion_ids']) for r in rows)
    flattened=Counter(x for r in rows for x in r['motion_ids'])
    psnr_keys=['psnr','workspace_roi_psnr','object_psnr','hand_activity_psnr']
    drops={k:summary['frozen_target'][k]-summary['correct_motion'][k] for k in psnr_keys}
    gains={k:summary['correct_motion'][k]-summary['held_initial'][k] for k in psnr_keys}
    shuffle_drop=summary['correct_motion']['hand_activity_psnr']-summary['shuffled_motion']['hand_activity_psnr']
    gates=protocol['evaluation']['pass']
    tests={'trained_causality':read(folder/'causality.json')['status']=='passed','all_pairs':len(rows)==protocol['evaluation']['pairs'],'packet_roundtrip':replay_report['passed'],
        'all_training_pairs_seen':training['all_pairs_seen'],
        'training_without_target_support_occurred':training['no_target_forcing_render_steps']>0,
        'full_and_region_drop':max(drops.values())<=gates['max_full_and_region_drop_vs_frozen_target_db'],
        'full_gain':gains['psnr']>=gates['min_full_gain_vs_held_initial_db'],
        'hand_gain':gains['hand_activity_psnr']>=gates['min_hand_gain_vs_held_initial_db'],
        'shuffle_control':shuffle_drop>=gates['min_hand_drop_when_motion_shuffled_db'],
        'distinct_motion_tuples':len(tuples)>=gates['min_distinct_motion_id_tuples']}
    result={'status':'complete','arm':arm,'motion_tokens':config['motion_tokens'],'training_steps':training['steps'],
        'pairs':len(rows),'elapsed_seconds':time.monotonic()-started,'binding':binding,'metrics':summary,
        'drop_vs_frozen_target_db':drops,'gain_vs_held_initial_db':gains,'shuffle_hand_drop_db':shuffle_drop,
        'unique_motion_tuples':len(tuples),'unique_motion_codes':len(flattened),
        'per_query_distinct_ids':[len({r['motion_ids'][k] for r in rows}) for k in range(config['motion_tokens'])],
        'top_motion_tuples':[{'ids':list(k),'count':v} for k,v in tuples.most_common(20)],
        'shuffle_changed_fraction':float(np.mean([r['shuffle_id_changed'] for r in rows])),
        'occupancy_precision':float(np.mean([r['occupancy_precision'] for r in rows])),
        'occupancy_recall':float(np.mean([r['occupancy_recall'] for r in rows])),
        'roundtrip':replay_report,'gates':tests,'quality_passed':all(tests.values()),
        'limitations':protocol['limitations']}
    eligible=[r for r in rows if r['context_frames']==10]
    history={'full10frame_pairs':len(eligible),
        'motion_id_changed_fraction':float(np.mean([r['history_changes_ids'] for r in eligible])),
        'full_psnr_gain_over_two_frame_context_db':float(np.mean([r['correct_motion']['psnr']-r['two_frame_context']['psnr'] for r in eligible])),
        'hand_psnr_gain_over_two_frame_context_db':float(np.mean([r['correct_motion']['hand_activity_psnr']-r['two_frame_context']['hand_activity_psnr'] for r in eligible])),
        'interpretation':'Post-hoc context intervention using same weights; positive render gains indicate observed history benefit, not an independent trained ablation.'}
    result.update(quantizer=config['quantizer'], vector_width=64, vocabulary=model.vocabulary, batch_size=training['batch_size'], pair_exposures=training['pair_exposures'],history_diagnostic=history)
    sequence_ids=torch.tensor([r['motion_ids'] for r in rows],dtype=torch.int64)[None]
    assert tuple(sequence_ids.shape)==(1,176,4)
    torch.save({'schema':'ar-action-ids-causal64-lfq-v2','ids':sequence_ids,'flattened_ids':sequence_ids.flatten(1),'binding':binding,'fps':fps,'vocabulary':model.vocabulary},output/'ar_motion_z.pt')
    write(output/'manifest.json',manifest);write(folder/'evaluation.json',result)
    print('EVAL_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--arm',required=True)
    args=parser.parse_args();evaluate(args.root,args.arm)

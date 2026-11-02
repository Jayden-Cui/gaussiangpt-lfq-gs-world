import json
import numpy as np
from PIL import Image,ImageDraw
import torch
from object_masks import color_group
from trial_utils import ROOT,FULL,sha,read,write,require,verify


def main():
    torch.set_num_threads(2);p=verify();out=ROOT/'masks';out.mkdir()
    manifest=read(FULL/'cache/MANIFEST.json');entry=manifest['frames']
    require(sha(FULL/'cache'/entry['file'])==entry['sha256'],'Input RGB cache changed')
    data=torch.load(FULL/'cache'/entry['file'],map_location='cpu',weights_only=True)
    masks=[];rows=[];tiles=[]
    for f in range(177):
        rgb=np.rint(data['samples'][f]['record']['real'].numpy().clip(0,1)*255).astype(np.uint8)
        per_frame=[];per_color=[]
        for spec in p['objects']:
            mask,stats=color_group(rgb,spec,p['mask_min_component_pixels']);per_frame.append(mask);per_color.append(dict(name=spec['name'],**stats))
        masks.append(np.stack(per_frame));rows.append(dict(frame=f,colors=per_color))
        rgb_image=Image.fromarray(rgb);masked=Image.fromarray(np.where(np.stack(per_frame).any(0)[...,None],rgb,0).astype(np.uint8))
        tile=Image.new('RGB',(448,146),'white');tile.paste(rgb_image.resize((224,126)),(0,20));tile.paste(masked.resize((224,126)),(224,20))
        ImageDraw.Draw(tile).text((4,3),f'{f:03d}  SOURCE / SELECTED COLOR REGIONS',fill='black');tiles.append(tile)
    mask_array=np.stack(masks);np.savez_compressed(out/'masks.npz',masks=mask_array,frame_indices=np.arange(177))
    contacts=[]
    for begin in range(0,177,36):
        page=Image.new('RGB',(4*448,9*146),'white')
        for i,tile in enumerate(tiles[begin:begin+36]):page.paste(tile,((i%4)*448,(i//4)*146))
        path=out/f'contact_{begin:03d}.png';page.save(path);contacts.append(dict(file=path.name,sha256=sha(path),start=begin,end=min(begin+35,176)))
    key=Image.new('RGB',(896,6*252),'white')
    for i,f in enumerate((35,38,44,123,126,129)):
        rgb=np.rint(data['samples'][f]['record']['real'].numpy().clip(0,1)*255).astype(np.uint8)
        key.paste(Image.fromarray(rgb),(0,i*252));key.paste(Image.fromarray(np.where(mask_array[f].any(0)[...,None],rgb,0).astype(np.uint8)),(448,i*252))
    key.save(out/'key_mask_review.png')
    write(out/'MASKS.json',dict(status='complete',protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256'),
        frames_cache_sha256=entry['sha256'],masks_sha256=sha(out/'masks.npz'),shape=list(mask_array.shape),rows=rows,
        contacts=contacts,key_contact_sha256=sha(out/'key_mask_review.png'),inference_input=False,
        note='Yellow includes both visible yellow bars in a fixed workspace ROI; green excludes the separate left triangle. Color proxies, not instance ground truth.'))
    print(json.dumps(dict(status='complete',frames=177,shape=list(mask_array.shape),mask_sha256=sha(out/'masks.npz'))))


if __name__=='__main__':main()

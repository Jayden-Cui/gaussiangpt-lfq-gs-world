"""Cross-source image metrics without inventing a missing teacher-view mask."""
import torch


@torch.no_grad()
def image_metrics_unmasked_teacher(images,window,sample):
    from core import image_metrics
    if (tuple(images.shape)!=(2,252,448,3) or tuple(sample['teacher_targets'].shape)!=(2,252,448,3)
            or tuple(sample['record']['real'].shape)!=(252,448,3)):
        raise ValueError('Expected two rendered views and bound source image')
    metrics=image_metrics(images[0],sample['record']['real'],window['roi'],sample['motion'])
    metrics['teacher_view1_full_mae']=float((images[1]-sample['teacher_targets'][1]).abs().mean())
    return dict(metrics=metrics,reference_view1_scope='full image; no teacher valid-region mask is available',
                roi_scope='fixed inherited image regions; not an object segmentation',
                comparison_note='Full-view teacher MAE is not the original masked-teacher MAE')

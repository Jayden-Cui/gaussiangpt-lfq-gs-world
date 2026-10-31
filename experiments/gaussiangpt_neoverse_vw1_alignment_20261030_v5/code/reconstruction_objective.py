"""Released image-objective primitives for the LFQ64 GS adaptation.

Contract: RGB tensors are signed [-1,1], video [B,10,3,252,448],
sampled image [B,3,252,448]. Native L1 and LPIPS keep that resolution;
only the discriminator branch is explicitly resized to 128x128.
The trainer owns D-then-G steps, freezing D during G, and LFQ auxiliary loss.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from vendor_gan import build_discriminator

NATIVE_HW = (252, 448)
DEFAULT_BACKBONE = Path('/home/amax/.cache/torch/hub/checkpoints/vgg16-397923af.pth')
DEFAULT_CALIBRATION = Path('/data/CJH/envs/gaussiangpt/lib/python3.10/site-packages/lpips/weights/v0.1/vgg.pth')
BACKBONE_SHA256 = '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'
CALIBRATION_SHA256 = 'a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868'


def _verified_file(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required local weights absent; downloading is disabled: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError(f'Local weight SHA256 mismatch: {path}')
    return path


def _images(images, joint_batch=False):
    if images.ndim != 4 or images.shape[1:] != (3, *NATIVE_HW):
        raise ValueError('Expected native RGB [B,3,252,448] in [-1,1]')
    if not images.is_floating_point():
        raise ValueError('Expected floating-point signed RGB')
    if joint_batch and images.shape[0] != 4:
        raise ValueError('D must receive four sampled frames jointly for group_size=4 minibatch stddev')


def _videos(fake, real):
    if fake.shape != real.shape or fake.ndim != 5 or fake.shape[1:] != (10, 3, *NATIVE_HW):
        raise ValueError('Both full videos must be [B,10,3,252,448], including the initial frame')
    if not fake.is_floating_point() or not real.is_floating_point():
        raise ValueError('Expected floating-point signed RGB')


def rgb01_to_signed(images):
    """Explicit range conversion; no clipping, resizing, or channel permutation."""
    return images * 2 - 1


def full_video_l1(fake, real):
    _videos(fake, real)
    return (fake - real).abs().mean()


def sample_frame_indices(batch_size, *, device=None, generator=None):
    """Original iid-normal argmax: each of the ten frames has probability 1/10.

    Draw on CPU like the released algorithm. Pass a CPU torch.Generator if
    controlling the RNG. D and G must call this separately; within G reuse
    the result for LPIPS and the adversarial prediction.
    """
    if batch_size < 1:
        raise ValueError('Empty batch')
    return torch.randn((batch_size, 10), generator=generator).topk(1, dim=-1).indices[:, 0].to(device)


def pick_video_frames(video, indices):
    if video.ndim != 5 or video.shape[1:] != (10, 3, *NATIVE_HW):
        raise ValueError('Expected the full ten-frame native video')
    if indices.shape != (video.shape[0],) or indices.dtype != torch.long:
        raise ValueError('Frame indices must be int64 [B]')
    if bool(((indices < 0) | (indices >= 10)).any()):
        raise ValueError('Frame index outside [0,9]')
    return video[torch.arange(video.shape[0], device=video.device), indices.to(video.device)]


class FrozenLPIPS(nn.Module):
    """Real v0.1 learned VGG LPIPS, fixed eval and frozen parameters.

    Intentional repairs relative to released vqperceptual.py: load learned
    lin0..4 weights correctly (upstream uses mismatched cos*/lins.* keys),
    and disable training-time Dropout. Existing files only; fail closed.
    """
    def __init__(self, backbone_path=DEFAULT_BACKBONE,
                 calibration_path=DEFAULT_CALIBRATION):
        super().__init__()
        backbone = _verified_file(backbone_path, BACKBONE_SHA256)
        calibration = _verified_file(calibration_path, CALIBRATION_SHA256)
        if backbone.name != 'vgg16-397923af.pth' or backbone.parent.name != 'checkpoints':
            raise ValueError('Backbone must use the checked torchvision cache filename/layout')
        import lpips
        old_hub = torch.hub.get_dir()
        try:
            # Initialization is serial; restore this process-local setting.
            torch.hub.set_dir(str(backbone.parent.parent))
            self.metric = lpips.LPIPS(net='vgg', version='0.1',
                model_path=str(calibration), pnet_rand=False, pnet_tune=False,
                use_dropout=True, eval_mode=True, verbose=False)
        finally:
            torch.hub.set_dir(old_hub)
        calibration_state = torch.load(calibration, map_location='cpu', weights_only=True)
        loaded = self.metric.state_dict()
        for key, value in calibration_state.items():
            if key not in loaded or not torch.equal(loaded[key].cpu(), value):
                raise RuntimeError(f'LPIPS calibration was not loaded: {key}')
        self.requires_grad_(False)
        self.train(False)
        self.weight_provenance = dict(backbone=str(backbone), calibration=str(calibration),
            backbone_sha256=BACKBONE_SHA256, calibration_sha256=CALIBRATION_SHA256)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, fake_frames, real_frames):
        _images(fake_frames)
        _images(real_frames)
        if fake_frames.shape != real_frames.shape:
            raise ValueError('LPIPS image shapes differ')
        return self.metric(fake_frames, real_frames.detach(), normalize=False).mean()


def discriminator_input(frames):
    _images(frames, joint_batch=True)
    return F.interpolate(frames, (128, 128), mode='bilinear', align_corners=False,
                         antialias=False)


def hinge_discriminator_loss(fake_scores, real_scores):
    return .5 * F.relu(1 + fake_scores).mean() + .5 * F.relu(1 - real_scores).mean()


def norm_to_one_gradient_penalty(real_images, real_scores):
    """Original real-only (||grad D(real)||2 - 1)^2; NOT R1."""
    gradients = torch.autograd.grad(outputs=real_scores, inputs=real_images,
        grad_outputs=torch.ones_like(real_scores), create_graph=True,
        retain_graph=True, only_inputs=True)[0]
    return ((gradients.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def discriminator_image_loss(discriminator, fake_frames, real_frames):
    """One D loss: B4 joint fake/real passes, hinge + GP(weight=1) each step.

    GP is differentiated w.r.t. the resized 128px input leaf, exactly the
    D's input coordinate system, not w.r.t. the upstream native image.
    """
    if fake_frames.shape != real_frames.shape:
        raise ValueError('Discriminator fake/real shapes differ')
    fake_input = discriminator_input(fake_frames.detach())
    real_input = discriminator_input(real_frames.detach()).requires_grad_(True)
    fake_scores = discriminator(fake_input)
    real_scores = discriminator(real_input)
    hinge = hinge_discriminator_loss(fake_scores, real_scores)
    penalty = norm_to_one_gradient_penalty(real_input, real_scores)
    return dict(loss=hinge + penalty, hinge=hinge, gradient_penalty=penalty,
                fake_scores=fake_scores, real_scores=real_scores)


def generator_adversarial_loss(fake_scores):
    return -fake_scores.mean()


def adaptive_adversarial_weight(nll_loss, adversarial_loss, last_layer):
    if not torch.is_tensor(last_layer) or not last_layer.requires_grad:
        raise ValueError('last_layer must be an explicitly selected trainable output-weight Tensor')
    nll_gradient = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    gan_gradient = torch.autograd.grad(adversarial_loss, last_layer, retain_graph=True)[0]
    return (.75 * (nll_gradient.norm() / (gan_gradient.norm() + 1e-4))
            .clamp(0, 1e4).detach())


def generator_image_loss(discriminator, perceptual, fake_video, real_video,
                         last_layer, *, frame_indices=None, generator=None):
    """All-ten native L1 + sampled native LPIPS + adaptive sampled GAN.

    Caller freezes D parameters after its update. Keep autograd enabled:
    gradients still flow through D to fake images and last_layer. LFQ
    conditional entropy is intentionally added once by the outer trainer.
    """
    _videos(fake_video, real_video)
    if fake_video.shape[0] != 4:
        raise ValueError('Four windows must be jointly represented for GAN minibatch stddev')
    if any(p.requires_grad for p in discriminator.parameters()):
        raise ValueError('Freeze discriminator parameters during the generator loss')
    if frame_indices is None:
        frame_indices = sample_frame_indices(4, device=fake_video.device, generator=generator)
    fake_frames = pick_video_frames(fake_video, frame_indices)
    real_frames = pick_video_frames(real_video, frame_indices)
    l1 = full_video_l1(fake_video, real_video.detach())
    lpips_loss = perceptual(fake_frames, real_frames)
    nll = l1 + lpips_loss
    gan = generator_adversarial_loss(discriminator(discriminator_input(fake_frames)))
    weight = adaptive_adversarial_weight(nll, gan, last_layer)
    return dict(loss=nll + weight * gan, l1=l1, lpips=lpips_loss, nll=nll,
                adversarial=gan, adaptive_weight=weight, frame_indices=frame_indices)

"""Encoder/IMU handoff geometry and hysteretic policy blending; no simulator imports."""
import math
import xml.etree.ElementTree as ET

import torch


def rotation(axis, angle):
    axis = axis / axis.norm()
    x, y, z = axis.unbind()
    zero = x * 0
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    identity = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return identity + angle.sin()[..., None, None] * skew + (1 - angle.cos())[..., None, None] * (skew @ skew)


class WheelLegFK:
    """URDF link transforms, including the offset between wheel link and cylinder center."""
    def __init__(self, urdf, joint_names, device='cpu'):
        root = ET.parse(urdf).getroot()
        self.device = device
        self.joints = []
        pending = list(root.findall('joint'))
        visited = {'base_Link'}
        self.wheels = []
        while pending:
            advanced = False
            for joint in pending[:]:
                parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
                if parent not in visited:
                    continue
                origin = joint.find('origin')
                xyz = torch.tensor([float(x) for x in origin.get('xyz', '0 0 0').split()], device=device)
                rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
                axes = torch.eye(3, device=device)
                orient = (rotation(axes[2], torch.tensor(rpy[2], device=device)) @
                          rotation(axes[1], torch.tensor(rpy[1], device=device)) @
                          rotation(axes[0], torch.tensor(rpy[0], device=device)))
                axis = torch.tensor([float(x) for x in joint.find('axis').get('xyz').split()], device=device)
                self.joints.append((parent, child, joint_names.index(joint.get('name')), xyz, orient, axis))
                visited.add(child)
                pending.remove(joint)
                advanced = True
            if not advanced:
                raise ValueError('URDF is not rooted at base_Link')
        for side in ('L', 'R'):
            link = root.find(f"link[@name='wheel_{side}_Link']")
            collision = link.find('collision')
            offset = torch.tensor([float(x) for x in collision.find('origin').get('xyz').split()], device=device)
            cylinder = collision.find('geometry/cylinder')
            self.wheels.append((side, offset, float(cylinder.get('radius')), float(cylinder.get('length')) / 2))

    def compute(self, q, gravity):
        n = len(q)
        frames = {'base_Link': (torch.eye(3, device=q.device).expand(n, -1, -1), torch.zeros(n, 3, device=q.device))}
        for parent, child, idx, xyz, orient, axis in self.joints:
            r, p = frames[parent]
            frames[child] = (r @ orient @ rotation(axis, q[:, idx]), p + (r @ xyz).squeeze(-1))
        centers, leg_lengths, heights, extensions, axles = [], [], [], [], []
        g = torch.nn.functional.normalize(gravity, dim=-1)
        for side, offset, radius, half_width in self.wheels:
            r, p = frames[f'wheel_{side}_Link']
            center = p + r @ offset
            leg = center - frames[f'hip_{side}_Link'][1]
            axle = r[:, :, 0]  # cylinder axis is link x, sign irrelevant
            vertical_axle = (axle * g).sum(-1).abs().clamp(0, 1)
            extent = radius * (1 - vertical_axle.square()).clamp_min(0).sqrt() + half_width * vertical_axle
            centers.append(center)
            leg_lengths.append(leg.norm(dim=-1))
            extensions.append((leg * g).sum(-1))
            heights.append((center * g).sum(-1) + extent)
            axles.append(axle)
        return dict(centers=torch.stack(centers, 1), lengths=torch.stack(leg_lengths, 1),
                    support_heights=torch.stack(heights, 1), extensions=torch.stack(extensions, 1),
                    axle_vertical=torch.stack([(a*g).sum(-1).abs() for a in axles], 1),
                    axle_alignment=(axles[0]*axles[1]).sum(-1).abs(),
                    tilt=torch.acos((-g[:, 2]).clamp(-1, 1)))


def handoff_conditions(geometry, nominal_lengths, angular_velocity, wheel_forces):
    """No world position, true base height, or simulator quaternion in the decision."""
    h, length = geometry['support_heights'], geometry['lengths']
    supported = (wheel_forces > 3.).all(-1)
    ready = ((geometry['tilt'] < math.radians(20)) & (h > .14).all(-1) & (h < .23).all(-1)
             & ((h[:, 0] - h[:, 1]).abs() < .035)
             & (length > .70 * nominal_lengths).all(-1) & (length < 1.35 * nominal_lengths).all(-1)
             & (geometry['extensions'] > .07).all(-1)
             & (geometry['axle_vertical'] < math.sin(math.radians(35))).all(-1)
             & (geometry['axle_alignment'] > math.cos(math.radians(45)))
             & ((geometry['centers'][:, 1, 0] - geometry['centers'][:, 0, 0]) > .08)
             & (angular_velocity.norm(dim=-1) < .8) & supported)
    unsafe = ((geometry['tilt'] > math.radians(40)) | (h < .10).any(-1)
              | (geometry['extensions'] < .04).any(-1) | ~supported)
    return ready, unsafe


class HandoffController:
    """RECOVERY -> BLEND -> LOCOMOTION; sustained instability smoothly reverses the blend."""
    def __init__(self, n, dt, device='cpu', hold=.3, blend=1., retreat=.3, ramp=1.):
        if min(dt, hold, blend, retreat, ramp) <= 0:
            raise ValueError('All handoff time constants must be positive')
        self.dt, self.hold, self.blend, self.retreat, self.ramp = dt, hold, blend, retreat, ramp
        self.phase = torch.zeros(n, device=device, dtype=torch.long)
        self.progress = torch.zeros(n, device=device)
        self.ready_time = torch.zeros(n, device=device)
        self.unsafe_time = torch.zeros(n, device=device)
        self.walk_time = torch.zeros(n, device=device)

    def reset(self, mask):
        for name in ('phase', 'progress', 'ready_time', 'unsafe_time', 'walk_time'):
            getattr(self, name)[mask] = 0

    @staticmethod
    def smooth(x):
        return x.square() * (3 - 2*x)

    def update(self, ready, unsafe):
        self.ready_time = torch.where(ready, self.ready_time + self.dt, 0.)
        self.unsafe_time = torch.where(unsafe, self.unsafe_time + self.dt, 0.)
        # Returning must complete before another attempt; no rapid toggling.
        begin = (self.phase == 0) & (self.ready_time >= self.hold - 1e-6)
        self.phase[begin] = 1
        retreat = ((self.phase == 1) | (self.phase == 2)) & (self.unsafe_time >= .15)
        self.phase[retreat] = 3
        self.progress = torch.where(self.phase == 1, self.progress + self.dt/self.blend, self.progress)
        self.progress = torch.where(self.phase == 3, self.progress - self.dt/self.retreat, self.progress).clamp(0., 1.)
        arrived = (self.phase == 1) & (self.progress >= 1.)
        self.phase[arrived] = 2
        returned = (self.phase == 3) & (self.progress <= 0.)
        self.phase[returned] = 0
        self.ready_time[returned] = 0
        # Ramp walking command separately: transition first to the standing locomotion policy.
        self.walk_time = torch.where(self.phase == 2, self.walk_time + self.dt,
                                     (self.walk_time - self.dt/self.retreat*self.ramp).clamp_min(0.)).clamp_max(self.ramp)
        return self.smooth(self.progress), self.smooth((self.walk_time/self.ramp).clamp(0., 1.))


class CheckpointPolicy:
    """Load only the actor and its matching encoder; play never uses the critic."""
    def __init__(self, path, device):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        def network(state, prefix, dims):
            layers = []
            for i, (a,b) in enumerate(zip(dims[:-1],dims[1:])):
                layers.append(torch.nn.Linear(a,b))
                if i < len(dims)-2:
                    layers.append(torch.nn.ELU())
            net = torch.nn.Sequential(*layers).to(device)
            net.load_state_dict({k.removeprefix(prefix):v for k,v in state.items() if k.startswith(prefix)}, strict=True)
            return net.eval()
        self.actor = network(checkpoint['model_state_dict'], 'actor.', [34,512,256,128,8])
        self.encoder = network(checkpoint['encoder_state_dict'], 'encoder.', [280,256,128,3])
        self.critic_input = checkpoint['model_state_dict']['critic.0.weight'].shape[1]

    def act(self, obs, history, command):
        return self.actor(torch.cat((self.encoder(history), obs, command), -1))


def blend_actions(recovery, locomotion, alpha, previous, scale, offset, low, high, rate, dt, limited):
    """Blend executable physical targets, then limit target slew during/after handoff."""
    rec = (recovery * scale + offset).clamp(low, high)
    loc = (locomotion * scale + offset).clamp(low, high)
    target = torch.lerp(rec, loc, alpha[:, None])
    last = (previous * scale + offset).clamp(low, high)
    slewed = last + (target-last).clamp(-rate*dt, rate*dt)
    target = torch.where(limited[:, None], slewed, target).clamp(low, high)
    blended = (target-offset)/scale
    # Recovery was trained with raw last_action observations, even when its
    # physical leg target saturates. Preserve that feedback until handoff starts.
    return torch.where(limited[:, None], blended, recovery)

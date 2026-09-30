"""Collision envelopes at a nonzero nominal pose, from the adapted USD's source URDF."""

import itertools
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def nominal_collision_corners(urdf_path, joint_positions):
    """Return conservative collision envelopes in base_Link coordinates (metres)."""
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()

    def origin(element):
        if element is None:
            return np.eye(4)
        rpy = [float(x) for x in element.get('rpy', '0 0 0').split()]
        transform = trimesh.transformations.euler_matrix(*rpy)
        transform[:3, 3] = [float(x) for x in element.get('xyz', '0 0 0').split()]
        return transform

    transforms = {'base_Link': np.eye(4)}
    pending = list(root.findall('joint'))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent not in transforms:
                continue
            transform = origin(joint.find('origin'))
            if joint.get('type') in ('revolute', 'continuous'):
                axis = [float(x) for x in joint.find('axis').get('xyz').split()]
                transform = transform @ trimesh.transformations.rotation_matrix(
                    joint_positions[joint.get('name')], axis)
            elif joint.get('type') != 'fixed':
                raise ValueError(f"Unsupported joint type: {joint.get('type')}")
            transforms[joint.find('child').get('link')] = transforms[parent] @ transform
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('URDF must be a connected tree rooted at base_Link')

    corners = []
    for link in root.findall('link'):
        for collision in link.findall('collision'):
            geometry = collision.find('geometry')
            mesh = geometry.find('mesh')
            cylinder = geometry.find('cylinder')
            box = geometry.find('box')
            if mesh is not None:
                points = trimesh.load(urdf_path.parent / mesh.get('filename'), force='mesh').convex_hull.vertices
                points = points * np.array([float(x) for x in mesh.get('scale', '1 1 1').split()])
            elif cylinder is not None:
                # A circumscribed 32-gon encloses the round wheel with <0.2 mm
                # error; a rotated square adds up to 15 mm of false clearance.
                radius = float(cylinder.get('radius')) / np.cos(np.pi / 32)
                angles = np.arange(32) * 2 * np.pi / 32
                half_length = float(cylinder.get('length')) / 2
                points = np.array([[radius * np.cos(a), radius * np.sin(a), z]
                                   for z in (-half_length, half_length) for a in angles])
            elif box is not None:
                half = np.array([float(x) for x in box.get('size').split()]) / 2
                bounds = np.stack((-half, half))
            else:
                raise ValueError('Unsupported collision geometry')
            if box is not None:
                points = np.array([[bounds[indices[i], i] for i in range(3)]
                                   for indices in itertools.product((0, 1), repeat=3)])
            transform = transforms[link.get('name')] @ origin(collision.find('origin'))
            corners.append(trimesh.transform_points(points, transform))
    if not corners:
        raise ValueError('No collision geometry in URDF')
    return np.concatenate(corners).astype(np.float32)

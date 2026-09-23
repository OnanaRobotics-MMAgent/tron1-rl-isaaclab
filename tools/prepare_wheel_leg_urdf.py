#!/usr/bin/env python3
"""Prepare the supplied wheel-leg URDF for the TRON1 IsaacLab task stack."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


JOINT_MAP = {
    "R_1": ("abad_R_Joint", "abad_R_Link"),
    "R_2": ("hip_R_Joint", "hip_R_Link"),
    "R_3": ("knee_R_Joint", "knee_R_Link"),
    "R_4": ("wheel_R_Joint", "wheel_R_Link"),
    "L_1": ("abad_L_Joint", "abad_L_Link"),
    "L_2": ("hip_L_Joint", "hip_L_Link"),
    "L_3": ("knee_L_Joint", "knee_L_Link"),
    "L_4": ("wheel_L_Joint", "wheel_L_Link"),
}

LEG_LIMITS = {
    "abad_L_Joint": (-0.383972, 1.396263),
    "abad_R_Joint": (-1.396263, 0.383972),
    "hip_L_Joint": (-1.012291, 0.0),
    "hip_R_Joint": (0.0, 1.012291),
    "knee_L_Joint": (-0.872665, 1.361357),
    "knee_R_Joint": (-1.361357, 0.872665),
}

WHEEL_MASS = 0.07
WHEEL_RADIUS = 0.0375
WHEEL_WIDTH = 0.035
WHEEL_COLLISION_CENTER_X = 0.0115

# A smooth, axis-aligned ellipsoid enclosing the base visual. Keep its upper
# hemisphere intact; clip only its lower hemisphere to the visual's convex
# hull. The visual STL is open, so its solid interior is not well defined.
BASE_ELLIPSOID_CENTER = (0.000068, -0.040214, 0.047032)
BASE_ELLIPSOID_RADII = (0.1045, 0.1145, 0.108)
BASE_ELLIPSOID_SUBDIVISIONS = 5
BASE_COLLISION_PREFIX = "base_ellipsoid_cut_"


def scale_inertial_mass(link: ET.Element, target_mass: float) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError(f"Link {link.attrib['name']} is missing inertial data")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass is None or inertia is None:
        raise ValueError(f"Link {link.attrib['name']} has incomplete inertial data")
    current_mass = float(mass.attrib["value"])
    if current_mass <= 0.0:
        raise ValueError(f"Link {link.attrib['name']} has non-positive mass")
    factor = target_mass / current_mass
    mass.attrib["value"] = f"{target_mass:.12g}"
    for name, value in inertia.attrib.items():
        inertia.attrib[name] = f"{float(value) * factor:.12g}"


def set_base_collision(link: ET.Element, part_names: list[str]) -> None:
    """Replace only the base collision geometry; leave visual and inertia intact."""
    if not part_names:
        raise ValueError("No convex base collision parts have been generated")
    for collision in link.findall("collision"):
        link.remove(collision)
    for name in part_names:
        collision = ET.SubElement(link, "collision", {"name": name.removesuffix(".stl")})
        ET.SubElement(collision, "origin", {
            "xyz": "0 0 0",
            "rpy": "0 0 0",
        })
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": f"meshes/{name}"})


def build_base_collision_parts(urdf_path: Path, mesh_dir: Path) -> list[str]:
    """Make two convex parts: upper ellipsoid and lower ellipsoid ∩ visual hull.

    Keep the equator at the ellipsoid center height. This is an appearance-
    driven cut and does not claim clearance over the complete joint travel.
    """
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull

    def clip_convex(points, plane):
        """Intersect a convex point hull with n·x + d <= 0."""
        distance = points @ plane[:3] + plane[3]
        if distance.max() <= 1e-10:
            return points
        if distance.min() >= -1e-10:
            raise ValueError("The base collision became empty while clipping")
        simplices = ConvexHull(points).simplices
        edges = np.unique(np.sort(np.concatenate(
            (simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]])), axis=1), axis=0)
        a, b = edges[:, 0], edges[:, 1]
        crossing = (distance[a] < -1e-10) & (distance[b] > 1e-10)
        crossing |= (distance[b] < -1e-10) & (distance[a] > 1e-10)
        a, b = a[crossing], b[crossing]
        fraction = distance[a] / (distance[a] - distance[b])
        intersections = points[a] + fraction[:, None] * (points[b] - points[a])
        return np.concatenate((points[distance <= 1e-10], intersections))

    center = np.asarray(BASE_ELLIPSOID_CENTER)
    radii = np.asarray(BASE_ELLIPSOID_RADII)
    ellipsoid = trimesh.creation.icosphere(subdivisions=BASE_ELLIPSOID_SUBDIVISIONS)
    points = ellipsoid.vertices * radii + center
    mid_z = center[2]
    upper = clip_convex(points, np.array([0, 0, -1, mid_z]))

    root = ET.parse(urdf_path).getroot()
    base = next(link for link in root.findall("link")
                if link.get("name") in ("base_link", "base_Link"))
    visual_filename = Path(base.find("visual/geometry/mesh").get("filename")).name
    visual_path = urdf_path.parent / "meshes" / visual_filename
    visual_hull = ConvexHull(trimesh.load_mesh(visual_path).vertices)
    # This hull is fully inside the enclosing ellipsoid. Consequently the
    # intersection of the lower ellipsoid and hull is just the clipped hull;
    # starting there also avoids ill-conditioned, almost-coplanar vertices.
    visual_vertices = visual_hull.points[visual_hull.vertices]
    if np.linalg.norm((visual_vertices - center) / radii, axis=1).max() > 1.0 + 1e-8:
        raise ValueError("The base visual hull protrudes outside the fitted ellipsoid")
    lower = clip_convex(visual_vertices, np.array([0, 0, 1, -mid_z]))

    parts = [
        trimesh.convex.convex_hull(upper),
        trimesh.convex.convex_hull(lower),
    ]
    mesh_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for index, part in enumerate(parts):
        name = f"{BASE_COLLISION_PREFIX}{index:03d}.stl"
        part.export(mesh_dir / name)
        names.append(name)
    for stale in mesh_dir.glob(f"{BASE_COLLISION_PREFIX}*.stl"):
        if stale.name not in names:
            stale.unlink()
    print(f"Generated {len(parts)} smooth convex base collision parts")
    return names


def set_wheel_inertial(link: ET.Element) -> None:
    """Use the analytic inertia of the supplied 70 g cylindrical wheel."""
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError(f"Link {link.attrib['name']} is missing inertial data")
    origin = inertial.find("origin")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if origin is None or mass is None or inertia is None:
        raise ValueError(f"Link {link.attrib['name']} has incomplete inertial data")
    i_axis = 0.5 * WHEEL_MASS * WHEEL_RADIUS**2
    i_radial = WHEEL_MASS * (3.0 * WHEEL_RADIUS**2 + WHEEL_WIDTH**2) / 12.0
    origin.attrib.update({"xyz": "0 0 0", "rpy": "0 0 0"})
    mass.attrib["value"] = f"{WHEEL_MASS:.12g}"
    inertia.attrib.update({
        "ixx": f"{i_axis:.12g}", "ixy": "0", "ixz": "0",
        "iyy": f"{i_radial:.12g}", "iyz": "0", "izz": f"{i_radial:.12g}",
    })


def prepare(input_path: Path, output_path: Path, part_names: list[str] | None = None) -> None:
    tree = ET.parse(input_path)
    root = tree.getroot()
    root.attrib["name"] = "tron1_wheel_leg"

    link_map = {"base_link": "base_Link"}
    link_map.update({old_link: new_link for _, new_link in JOINT_MAP.values() for old_link in []})
    link_map.update({old_name: new_link for old_name, (_, new_link) in JOINT_MAP.items()})

    for link in root.findall("link"):
        old_name = link.attrib["name"]
        new_name = link_map.get(old_name, old_name)
        link.attrib["name"] = new_name
        for mesh in link.findall("./visual/geometry/mesh") + link.findall("./collision/geometry/mesh"):
            filename = mesh.attrib.get("filename", "")
            mesh.attrib["filename"] = f"meshes/{Path(filename).name}"

    links = {link.attrib["name"]: link for link in root.findall("link")}
    scale_inertial_mass(links["base_Link"], 2.4)
    if part_names is None:
        mesh_dir = (Path(__file__).resolve().parents[1] /
                    "exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/meshes")
        part_names = sorted(path.name for path in mesh_dir.glob(f"{BASE_COLLISION_PREFIX}*.stl"))
    set_base_collision(links["base_Link"], part_names)
    left_wheel_inertial = links["wheel_L_Link"].find("inertial")
    right_wheel_inertial = links["wheel_R_Link"].find("inertial")
    if left_wheel_inertial is None or right_wheel_inertial is None:
        raise ValueError("Both wheel links must contain inertial data")
    for element_name in ("origin", "inertia"):
        source = right_wheel_inertial.find(element_name)
        target = left_wheel_inertial.find(element_name)
        if source is not None and target is not None:
            target.attrib.update(source.attrib)
    scale_inertial_mass(links["wheel_L_Link"], 0.07)
    scale_inertial_mass(links["wheel_R_Link"], 0.07)
    for side in ("L", "R"):
        leg_links = [links[f"{part}_{side}_Link"] for part in ("abad", "hip", "knee")]
        total_mass = sum(float(link.find("inertial/mass").attrib["value"]) for link in leg_links)
        scale = 1.8 / total_mass
        for link in leg_links:
            scale_inertial_mass(link, float(link.find("inertial/mass").attrib["value"]) * scale)
    set_wheel_inertial(links["wheel_L_Link"])
    set_wheel_inertial(links["wheel_R_Link"])

    # The STL bounds are [-29, 6] mm on the left and [-6, 29] mm on the
    # right along the axle, so center each contact cylinder on its visual.
    for side, center_x in (("L", -WHEEL_COLLISION_CENTER_X), ("R", WHEEL_COLLISION_CENTER_X)):
        link = links[f"wheel_{side}_Link"]
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            if origin is None:
                origin = ET.SubElement(collision, "origin")
            origin.attrib.update({"xyz": f"{center_x:.12g} 0 0", "rpy": "0 1.570796326795 0"})
            geometry = collision.find("geometry")
            if geometry is None:
                geometry = ET.SubElement(collision, "geometry")
            for child in list(geometry):
                geometry.remove(child)
            ET.SubElement(geometry, "cylinder", {
                "radius": f"{WHEEL_RADIUS:.12g}",
                "length": f"{WHEEL_WIDTH:.12g}",
            })

    for joint in root.findall("joint"):
        old_name = joint.attrib["name"]
        if old_name not in JOINT_MAP:
            raise ValueError(f"Unexpected joint in supplied URDF: {old_name}")
        new_name, _ = JOINT_MAP[old_name]
        joint.attrib["name"] = new_name
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {old_name} is missing parent or child")
        parent.attrib["link"] = link_map[parent.attrib["link"]]
        child.attrib["link"] = link_map[child.attrib["link"]]

        limit = joint.find("limit")
        if old_name.endswith("_4"):
            joint.attrib["type"] = "continuous"
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            limit.attrib.clear()
            limit.attrib.update({"effort": "8.0", "velocity": "20.0"})
        else:
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            new_name = joint.attrib["name"]
            lower, upper = LEG_LIMITS[new_name]
            limit.attrib.update({"lower": str(lower), "upper": str(upper)})
            limit.attrib.update({"effort": "20.0", "velocity": "12.0"})

    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rebuild-base-collision", action="store_true",
                        help="Generate upper ellipsoid and lower visual-hull-clipped collision meshes")
    args = parser.parse_args()
    output = args.output.resolve()
    part_names = None
    if args.rebuild_base_collision:
        part_names = build_base_collision_parts(
            args.input.resolve(), output.parent / "meshes")
    prepare(args.input.resolve(), output, part_names)
    print(f"Prepared URDF: {args.output.resolve()}")


if __name__ == "__main__":
    main()

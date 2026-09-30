#!/usr/bin/env python3
"""Audit and prepare Motor35 without applying the OLD robot's mass/axis edits.

Run with Isaac Sim's python.sh (numpy, scipy, trimesh). Outputs are generated
under bipedal_locomotion_motor35/assets/urdf; original CAD/URDF files are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import trimesh

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "exts/bipedal_locomotion/bipedal_locomotion_motor35/assets/config/motor35_parameters.py"
module_spec = importlib.util.spec_from_file_location("motor35_parameters", SPEC)
motor = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(motor)
SOURCE = ROOT / "exts/bipedal_locomotion/bipedal_locomotion_motor35/assets/source/Motor35_URDF/轮腿总装/urdf/轮腿总装（角度限位调整版本）.urdf"
CAD = ROOT / "exts/bipedal_locomotion/bipedal_locomotion_motor35/assets/source/Motor35_URDF/转动惯量.txt"


def origin(element):
    if element is None:
        return np.eye(4)
    tf = trimesh.transformations.euler_matrix(*map(float, element.get("rpy", "0 0 0").split()))
    tf[:3, 3] = list(map(float, element.get("xyz", "0 0 0").split()))
    return tf


def fk(robot, positions):
    children = {j.find("child").get("link") for j in robot.findall("joint")}
    roots = {l.get("name") for l in robot.findall("link")} - children
    if len(roots) != 1:
        raise ValueError("URDF must have exactly one root")
    transforms = {roots.pop(): np.eye(4)}
    pending = list(robot.findall("joint"))
    while pending:
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            tf = origin(joint.find("origin"))
            if joint.get("type") in ("revolute", "continuous"):
                tf = tf @ trimesh.transformations.rotation_matrix(
                    positions.get(joint.get("name"), 0.0),
                    list(map(float, joint.find("axis").get("xyz").split())))
            elif joint.get("type") != "fixed":
                raise ValueError("Unsupported joint type")
            child = joint.find("child").get("link")
            if child in transforms:
                raise ValueError("URDF is not a tree")
            transforms[child] = transforms[parent] @ tf
            pending.remove(joint)
        else:
            if pending and not any(j.find("parent").get("link") in transforms for j in pending):
                raise ValueError("Disconnected URDF / cycle")
    return transforms


def inertia_matrix(link):
    a = {k: float(v) for k, v in link.find("inertial/inertia").attrib.items()}
    return np.array([[a["ixx"], a["ixy"], a["ixz"]],
                     [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]])


def cad_properties(path):
    text = path.read_text()
    blocks = re.split(r"(?m)^\s*(Base_Link|hip_[RL]|thigh_[RL]|calf_[RL]|wheel_[RL])\s*$", text)
    result = {}
    for name, block in zip(blocks[1::2], blocks[2::2]):
        mass = float(re.search(r"质量\s*=\s*([\d.]+)", block)[1])
        com = np.array([float(re.search(rf"\b{a}\s*=\s*([-\d.]+)", block)[1]) for a in "XYZ"]) * 1e-3
        tensor = np.array([[float(re.search(rf"L{a}{b}\s*=\s*([-\d.]+)", block)[1])
                            for b in "xyz"] for a in "xyz"]) * 1e-6
        result[name] = (mass, com, tensor)
    return result


def audit(source=SOURCE, cad=CAD):
    robot = ET.parse(source).getroot()
    assert {j.get("name") for j in robot.findall("joint")} == set(motor.JOINT_MAP)
    assert {l.get("name") for l in robot.findall("link")} == set(motor.LINK_MAP)
    zero_tf = fk(robot, {})
    ref = cad_properties(cad)
    links, meshes, total_mass = {}, {}, 0.0
    for link in robot.findall("link"):
        name = link.get("name")
        mass = float(link.find("inertial/mass").get("value"))
        tensor = inertia_matrix(link)
        principal = np.linalg.eigvalsh(tensor)
        if mass <= 0 or not np.isfinite(tensor).all() or principal[0] <= 0 or principal[-1] > sum(principal[:2]):
            raise ValueError(f"Invalid mass/inertia: {name}")
        inertial_tf = zero_tf[name] @ origin(link.find("inertial/origin"))
        cad_name = "Base_Link" if name == "base_link" else name.split("_", 1)[1]
        cmass, ccom, ctensor = ref[cad_name]
        np.testing.assert_allclose(mass, cmass, atol=1e-9)
        np.testing.assert_allclose(inertial_tf[:3, 3], ccom, atol=1e-6)
        np.testing.assert_allclose(inertial_tf[:3, :3] @ tensor @ inertial_tf[:3, :3].T, ctensor, atol=1e-9)
        mesh_path = source.parent.parent / "meshes" / Path(link.find("collision/geometry/mesh").get("filename")).name
        mesh = trimesh.load_mesh(mesh_path)
        meshes[name] = mesh
        links[name] = {"mass_kg": mass, "principal_inertia_kg_m2": principal.tolist(),
                       "mesh_bounds_m": mesh.bounds.tolist(), "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest()}
        total_mass += mass
    joints = {}
    positions = {src: motor.NOMINAL_JOINT_POS[dst] for src, dst in motor.JOINT_MAP.items()}
    for joint in robot.findall("joint"):
        name = joint.get("name")
        limits = {k: float(v) for k, v in joint.find("limit").attrib.items()}
        np.testing.assert_allclose(limits["effort"], motor.PEAK_TORQUE_NM)
        np.testing.assert_allclose(limits["velocity"], motor.NO_LOAD_SPEED_RAD_S)
        if joint.get("type") == "revolute":
            low, high = limits["lower"], limits["upper"]
            assert low < positions[name] < high
            assert low + .05 * (high - low) < positions[name] < high - .05 * (high - low)
        elif joint.get("type") != "continuous":
            raise ValueError(f"Unexpected type: {name}")
        joints[name] = {"mapped_name": motor.JOINT_MAP[name], "axis": joint.find("axis").get("xyz"), **limits}
    tf = fk(robot, positions)
    com = sum(float(l.find("inertial/mass").get("value")) *
              (tf[l.get("name")] @ origin(l.find("inertial/origin")))[:3, 3]
              for l in robot.findall("link")) / total_mass
    wheel = (tf["Link_wheel_R"][:3, 3] + tf["Link_wheel_L"][:3, 3]) / 2
    np.testing.assert_allclose(motor.WHEEL_RADIUS_M - wheel[2], motor.STAND_HEIGHT_M, atol=1e-7)
    np.testing.assert_allclose(com[0], wheel[0], atol=1e-7)
    # Detect WHY naive convex-hull import is unsuitable. The radius below is
    # an inscribed intersection sphere, NOT penetration depth / exact CAD contact.
    hulls = {n: ConvexHull(trimesh.transform_points(m.convex_hull.vertices, tf[n])).equations
             for n, m in meshes.items()}
    adjacent = {frozenset((j.find("parent").get("link"), j.find("child").get("link"))) for j in robot.findall("joint")}
    overlaps = []
    for a, b in itertools.combinations(hulls, 2):
        if frozenset((a, b)) in adjacent:
            continue
        eq = np.concatenate((hulls[a], hulls[b]))
        solution = linprog([0., 0., 0., -1.], A_ub=np.c_[eq[:, :3], np.ones(len(eq))],
                           b_ub=-eq[:, 3], bounds=[(None, None)] * 4, method="highs")
        if solution.success and solution.x[3] > 1e-5:
            overlaps.append({"links": [a, b], "intersection_sphere_radius_m": float(solution.x[3])})
    wheel_bounds = links["Link_wheel_R"]["mesh_bounds_m"]
    np.testing.assert_allclose(np.array(wheel_bounds)[:, [0, 2]], [[-.04, -.04], [.04, .04]], atol=5e-5)
    return {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "cad_sha256": hashlib.sha256(cad.read_bytes()).hexdigest(),
            "profile_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
            "total_mass_kg": total_mass, "links": links, "joints": joints,
            "cad_inertia_and_com_match": True, "nominal_joint_pos": motor.NOMINAL_JOINT_POS,
            "nominal_root_height_m": motor.STAND_HEIGHT_M, "nominal_com_base_m": com.tolist(),
            "wheel_center_base_m": wheel.tolist(), "forward_axis": "x",
            "naive_convex_hull_overlaps": overlaps,
            "warnings": ["CSV is stale; URDF + CAD COM tensor in SI are authoritative.",
                         "Use convex decomposition; convex-hull intersections are NOT proof of CAD interference.",
                         "Simulation PD gains and torque-speed approximation still need hardware calibration."]}


def prepare(output, source=SOURCE):
    report = audit(source)
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("name", "Motor35")
    # Include mesh/profile identity so Isaac's URDF cache invalidates on CAD edits.
    fingerprint = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    root.insert(0, ET.Comment(f" Motor35 source/profile/mesh fingerprint: {fingerprint} "))
    output.mkdir(parents=True, exist_ok=True)
    (output / "meshes").mkdir(exist_ok=True)
    for link in root.findall("link"):
        name = motor.LINK_MAP[link.get("name")]
        link.set("name", name)
        for mesh in link.findall(".//mesh"):
            filename = Path(mesh.get("filename")).name
            source_mesh = source.parent.parent / "meshes" / filename
            # ASCII asset filenames avoid importer encoding and ROS package resolution issues.
            destination = output / "meshes" / f"{name}.stl"
            shutil.copyfile(source_mesh, destination)
            mesh.set("filename", f"meshes/{name}.stl")
        if name.startswith("wheel_"):
            collision = link.find("collision")
            collision.clear()
            side = 1 if "_R_" in name else -1
            ET.SubElement(collision, "origin", xyz=f"0 {side * motor.WHEEL_CENTER_Y_M} 0", rpy="1.5707963267948966 0 0")
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "cylinder", radius=str(motor.WHEEL_RADIUS_M), length=str(motor.WHEEL_WIDTH_M))
    for joint in root.findall("joint"):
        joint.set("name", motor.JOINT_MAP[joint.get("name")])
        for tag in ("parent", "child"):
            element = joint.find(tag)
            element.set("link", motor.LINK_MAP[element.get("link")])
    ET.indent(tree, space="  ")
    tree.write(output / "robot.urdf", encoding="utf-8", xml_declaration=True)
    report["prepared_sha256"] = hashlib.sha256((output / "robot.urdf").read_bytes()).hexdigest()
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "exts/bipedal_locomotion/bipedal_locomotion_motor35/assets/urdf")
    args = parser.parse_args()
    report = prepare(args.output.resolve())
    print(json.dumps({k: report[k] for k in ("total_mass_kg", "cad_inertia_and_com_match", "forward_axis", "naive_convex_hull_overlaps")}, indent=2))
    print(f"Prepared {args.output}/robot.urdf; detailed audit: {args.output}/audit.json")

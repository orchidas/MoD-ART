#!/usr/bin/env python3
"""Generate MoD-ART mesh.obj/mesh.mtl for ERTD two-room coupled geometry.

The geometry follows test/test_ertd.py:
  - Room 1 (hallway): x in [0, 4.5], y in [0, 18.0], z in [0, 2.8]
  - Room 2 (meeting): x in [4.5, 9.1], y in [7.0, 13.6], z in [0, 2.8]
  - Door aperture on shared wall x=4.5:
      y in [9.5, 10.5], z in [0, 2.0]

Patching strategy:
  - Axis-aligned walls are split into rectangular cells with target cell area.
  - Each cell is one ART patch (two triangles).
  - The shared wall is generated for both rooms, excluding the aperture opening.
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class MaterialStyle:
    """Visual/material metadata used to emit MTL patch definitions.

    Attributes:
        name (str): Acoustic/semantic material tag used in OBJ/MTL names
            as `Patch_{i}_Mat_{name}`.
        kd (tuple[float, float, float]): Diffuse RGB color in [0, 1] for
            `Kd` entries in the MTL file.
    """

    name: str
    kd: Tuple[float, float, float]


class ObjBuilder:
    """Incrementally builds an OBJ mesh and matching patch assignments.

    Attributes:
        vertices (list[Vec3]): Unique vertex positions stored in OBJ order
            (1-indexed when referenced by faces).
        vertex_index (dict[tuple[int, int, int], int]): Quantized coordinate
            key -> OBJ vertex index map used for deduplication.
        patches (list[tuple[str, list[tuple[int, int, int]]]]): Per-patch
            entries as `(material_tag, triangles)`.
    """

    def __init__(self) -> None:
        """Initialize an empty mesh/patch container.

        Returns:
            None
        """
        self.vertices: List[Vec3] = []
        self.vertex_index: Dict[Tuple[int, int, int], int] = {}
        self.patches: List[Tuple[str, List[Tuple[int, int, int]]]] = []

    def add_vertex(self, p: Vec3) -> int:
        """Add a vertex if not present and return its OBJ index.

        Args:
            p (tuple[float, float, float]): Vertex position `(x, y, z)` in
                meters.

        Returns:
            int: 1-based OBJ vertex index of `p` (existing or newly created).
        """
        key = (round(p[0] * 1_000_000), round(p[1] * 1_000_000),
               round(p[2] * 1_000_000))
        if key in self.vertex_index:
            return self.vertex_index[key]
        self.vertices.append(p)
        idx = len(self.vertices)  # OBJ is 1-indexed
        self.vertex_index[key] = idx
        return idx

    def add_patch_from_quad(self, material_tag: str, p00: Vec3, p10: Vec3,
                            p11: Vec3, p01: Vec3,
                            desired_normal: Vec3) -> None:
        """Add one rectangular patch as two triangles, with controlled winding.

        The quad vertices are assumed to be ordered around the perimeter.
        Triangle winding is flipped if needed so the first triangle normal
        aligns with `desired_normal`.

        Args:
            material_tag (str): Material name suffix used in
                `Patch_{i}_Mat_{material_tag}`.
            p00 (tuple[float, float, float]): Corner 0 position.
            p10 (tuple[float, float, float]): Corner 1 position.
            p11 (tuple[float, float, float]): Corner 2 position.
            p01 (tuple[float, float, float]): Corner 3 position.
            desired_normal (tuple[float, float, float]): Target unit normal
                direction for the patch.

        Returns:
            None
        """
        i00 = self.add_vertex(p00)
        i10 = self.add_vertex(p10)
        i11 = self.add_vertex(p11)
        i01 = self.add_vertex(p01)

        n = _normal_from_indices(self.vertices, (i00, i10, i11))
        if _dot(n, desired_normal) < 0:
            i10, i01 = i01, i10

        tri_1 = (i00, i10, i11)
        tri_2 = (i00, i11, i01)
        self.patches.append((material_tag, [tri_1, tri_2]))

    def write_obj(self, path: Path) -> None:
        """Write a Wavefront OBJ file with patch-based `usemtl` blocks.

        Args:
            path (pathlib.Path): Output path for `mesh.obj`.

        Returns:
            None
        """
        with path.open("w", encoding="utf-8") as f:
            f.write("mtllib mesh.mtl\n\n")
            f.write("################################ Vertices\n\n")
            for i, (x, y, z) in enumerate(self.vertices, start=1):
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}                  # {i}\n")

            f.write("\n################################ Faces\n\n")
            for patch_idx, (mat_tag, tris) in enumerate(self.patches, start=1):
                f.write(f"usemtl Patch_{patch_idx}_Mat_{mat_tag}\n")
                for (a, b, c) in tris:
                    f.write(f"f {a} {b} {c}\n")
                f.write("\n")

    def write_mtl(self, path: Path, patch_materials: Sequence[MaterialStyle],
                  patch_tags: Sequence[str]) -> None:
        """Write MTL entries corresponding to all generated patches.

        Args:
            path (pathlib.Path): Output path for `mesh.mtl`.
            patch_materials (Sequence[MaterialStyle]): Available material
                definitions (name + RGB).
            patch_tags (Sequence[str]): Material name for each patch, in patch
                index order.

        Returns:
            None
        """
        kd_by_name = {m.name: m.kd for m in patch_materials}
        with path.open("w", encoding="utf-8") as f:
            for patch_idx, tag in enumerate(patch_tags, start=1):
                kd = kd_by_name[tag]
                f.write(f"newmtl Patch_{patch_idx}_Mat_{tag}\n")
                f.write(f"Kd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}\n")


def _dot(a: Vec3, b: Vec3) -> float:
    """Compute 3D dot product.

    Args:
        a (tuple[float, float, float]): First vector.
        b (tuple[float, float, float]): Second vector.

    Returns:
        float: Dot product `a·b`.
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normal_from_indices(vertices: Sequence[Vec3], tri: Tuple[int, int,
                                                              int]) -> Vec3:
    """Compute a unit face normal from OBJ-style triangle indices.

    Args:
        vertices (Sequence[tuple[float, float, float]]): Vertex list in OBJ
            order (1-based indexing expected by `tri`).
        tri (tuple[int, int, int]): Triangle vertex indices (1-based).

    Returns:
        tuple[float, float, float]: Unit normal vector.

    Raises:
        ValueError: If the triangle is degenerate (zero area).
    """
    a = vertices[tri[0] - 1]
    b = vertices[tri[1] - 1]
    c = vertices[tri[2] - 1]
    e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    n = (
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    )
    mag = math.sqrt(_dot(n, n))
    if mag == 0:
        raise ValueError(
            "Degenerate triangle encountered while building mesh.")
    return (n[0] / mag, n[1] / mag, n[2] / mag)


def _best_grid_counts(len_u: float, len_v: float,
                      target_area: float) -> Tuple[int, int]:
    """Pick grid resolution `(nu, nv)` for a rectangle near target patch area.

    Args:
        len_u (float): Rectangle size along local `u` axis (meters).
        len_v (float): Rectangle size along local `v` axis (meters).
        target_area (float): Desired area per patch cell (m^2).

    Returns:
        tuple[int, int]: Number of subdivisions along `u` and `v`.
    """
    target_side = math.sqrt(target_area)
    raw_u = len_u / target_side
    raw_v = len_v / target_side

    cand_u = sorted({max(1, math.floor(raw_u)), max(1, math.ceil(raw_u))})
    cand_v = sorted({max(1, math.floor(raw_v)), max(1, math.ceil(raw_v))})

    best = (1, 1)
    best_err = float("inf")
    for nu in cand_u:
        for nv in cand_v:
            cell_area = (len_u * len_v) / (nu * nv)
            err = abs(cell_area - target_area)
            if err < best_err:
                best_err = err
                best = (nu, nv)
    return best


def _linspace(a: float, b: float, n: int) -> List[float]:
    """Return `n+1` evenly spaced points from `a` to `b`.

    Args:
        a (float): Start value.
        b (float): End value.
        n (int): Number of intervals.

    Returns:
        list[float]: Monotonic point list from `a` to `b` inclusive.
    """
    if n <= 0:
        return [a, b]
    step = (b - a) / n
    return [a + i * step for i in range(n + 1)]


def _emit_rect_grid(
    builder: ObjBuilder,
    mat_tag: str,
    axis: str,
    const_val: float,
    u_bounds: Tuple[float, float],
    v_bounds: Tuple[float, float],
    desired_normal: Vec3,
    patch_area: float,
) -> None:
    """Emit a tessellated rectangular wall aligned to x, y, or z plane.

    Args:
        builder (ObjBuilder): Mesh builder to receive generated patches.
        mat_tag (str): Material name used for all generated patches.
        axis (str): Plane axis fixed to `const_val`; one of `"x"`, `"y"`,
            `"z"`.
        const_val (float): Constant coordinate of the plane.
        u_bounds (tuple[float, float]): Min/max bounds on first in-plane axis.
        v_bounds (tuple[float, float]): Min/max bounds on second in-plane axis.
        desired_normal (tuple[float, float, float]): Desired outward/inward
            patch normal direction.
        patch_area (float): Target rectangular cell area in m^2.

    Returns:
        None

    Raises:
        ValueError: If `axis` is not one of `"x"`, `"y"`, `"z"`.
    """
    u0, u1 = u_bounds
    v0, v1 = v_bounds
    len_u = abs(u1 - u0)
    len_v = abs(v1 - v0)
    nu, nv = _best_grid_counts(len_u, len_v, patch_area)
    u_points = _linspace(u0, u1, nu)
    v_points = _linspace(v0, v1, nv)

    for iu in range(nu):
        for iv in range(nv):
            ua, ub = u_points[iu], u_points[iu + 1]
            va, vb = v_points[iv], v_points[iv + 1]

            if axis == "x":
                p00 = (const_val, ua, va)
                p10 = (const_val, ub, va)
                p11 = (const_val, ub, vb)
                p01 = (const_val, ua, vb)
            elif axis == "y":
                p00 = (ua, const_val, va)
                p10 = (ub, const_val, va)
                p11 = (ub, const_val, vb)
                p01 = (ua, const_val, vb)
            elif axis == "z":
                p00 = (ua, va, const_val)
                p10 = (ub, va, const_val)
                p11 = (ub, vb, const_val)
                p01 = (ua, vb, const_val)
            else:
                raise ValueError(f"Unsupported axis: {axis}")

            builder.add_patch_from_quad(mat_tag, p00, p10, p11, p01,
                                        desired_normal)


def generate_mesh(output_dir: Path, patch_area: float,
                  which_ertd: int) -> None:
    """Generate `mesh.obj` and `mesh.mtl` for the ERTD coupled-room geometry.

    Args:
        output_dir (pathlib.Path): Destination directory where files are
            written.
        patch_area (float): Target area (m^2) per rectangular surface patch.
        which_ertd (int): which ERTD dataset are we simulating?

    Returns:
        None

    Raises:
        ValueError: If `patch_area <= 0`.
    """
    if patch_area <= 0.0:
        raise ValueError("patch_area must be > 0")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    b = ObjBuilder()

    hallway_left = MaterialStyle("HallwayLeftWall", tuple(args.hallway_kd))
    hallway_right = MaterialStyle("HallwayRightWall", tuple(args.hallway_kd))
    hallway_floor = MaterialStyle("HallwayFloor", tuple(args.hallway_kd))
    hallway_ceiling = MaterialStyle("HallwayCeiling", tuple(args.hallway_kd))
    hallway_front = MaterialStyle("HallwayFrontWall", tuple(args.hallway_kd))
    hallway_back = MaterialStyle("HallwayBackWall", tuple(args.hallway_kd))

    meeting_left = MaterialStyle("MeetingLeftWall", tuple(args.meeting_kd))
    meeting_right = MaterialStyle("MeetingRightWall", tuple(args.meeting_kd))
    meeting_floor = MaterialStyle("MeetingFloor", tuple(args.meeting_kd))
    meeting_ceiling = MaterialStyle("MeetingCeiling", tuple(args.meeting_kd))
    meeting_front = MaterialStyle("MeetingFrontWall", tuple(args.meeting_kd))
    meeting_back = MaterialStyle("MeetingBackWall", tuple(args.meeting_kd))

    # Room 1 (hallway)
    _emit_rect_grid(b, hallway_floor.name, "z", 0.0, (0.0, 4.5), (0.0, 18.0),
                    (0.0, 0.0, 1.0), patch_area)
    _emit_rect_grid(b, hallway_ceiling.name, "z", 2.8, (0.0, 4.5), (0.0, 18.0),
                    (0.0, 0.0, -1.0), patch_area)
    _emit_rect_grid(b, hallway_left.name, "x", 0.0, (0.0, 18.0), (0.0, 2.8),
                    (1.0, 0.0, 0.0), patch_area)
    _emit_rect_grid(b, hallway_back.name, "y", 0.0, (0.0, 4.5), (0.0, 2.8),
                    (0.0, 1.0, 0.0), patch_area)
    _emit_rect_grid(b, hallway_front.name, "y", 18.0, (0.0, 4.5), (0.0, 2.8),
                    (0.0, -1.0, 0.0), patch_area)

    # Room 1: shared wall exterior segments (outside overlap with room 2)
    _emit_rect_grid(b, hallway_right.name, "x", 4.5, (0.0, 7.0), (0.0, 2.8),
                    (-1.0, 0.0, 0.0), patch_area)
    _emit_rect_grid(b, hallway_right.name, "x", 4.5, (13.6, 18.0), (0.0, 2.8),
                    (-1.0, 0.0, 0.0), patch_area)

    # Room 1: shared wall overlap with aperture cutout (y in [7,13.6], z in [0,2.8] minus door)
    _emit_rect_grid(b, hallway_right.name, "x", 4.5, (7.0, 9.5), (0.0, 2.8),
                    (-1.0, 0.0, 0.0), patch_area)
    _emit_rect_grid(b, hallway_right.name, "x", 4.5, (9.5, 10.5), (2.0, 2.8),
                    (-1.0, 0.0, 0.0), patch_area)
    _emit_rect_grid(b, hallway_right.name, "x", 4.5, (10.5, 13.6), (0.0, 2.8),
                    (-1.0, 0.0, 0.0), patch_area)

    # Room 2 (meeting room)
    _emit_rect_grid(b, meeting_floor.name, "z", 0.0, (4.5, 9.1), (7.0, 13.6),
                    (0.0, 0.0, 1.0), patch_area)
    _emit_rect_grid(b, meeting_ceiling.name, "z", 2.8, (4.5, 9.1), (7.0, 13.6),
                    (0.0, 0.0, -1.0), patch_area)
    _emit_rect_grid(b, meeting_right.name, "x", 9.1, (7.0, 13.6), (0.0, 2.8),
                    (-1.0, 0.0, 0.0), patch_area)
    _emit_rect_grid(b, meeting_back.name, "y", 7.0, (4.5, 9.1), (0.0, 2.8),
                    (0.0, 1.0, 0.0), patch_area)
    _emit_rect_grid(b, meeting_front.name, "y", 13.6, (4.5, 9.1), (0.0, 2.8),
                    (0.0, -1.0, 0.0), patch_area)

    # Room 2: shared wall overlap with aperture cutout (opposite normal)
    _emit_rect_grid(b, meeting_left.name, "x", 4.5, (7.0, 9.5), (0.0, 2.8),
                    (1.0, 0.0, 0.0), patch_area)
    _emit_rect_grid(b, meeting_left.name, "x", 4.5, (9.5, 10.5), (2.0, 2.8),
                    (1.0, 0.0, 0.0), patch_area)
    _emit_rect_grid(b, meeting_left.name, "x", 4.5, (10.5, 13.6), (0.0, 2.8),
                    (1.0, 0.0, 0.0), patch_area)

    patch_tags = [mat for mat, _ in b.patches]
    b.write_obj(output_dir / "mesh.obj")
    b.write_mtl(
        output_dir / "mesh.mtl",
        [
            hallway_floor,
            hallway_ceiling,
            hallway_left,
            hallway_right,
            hallway_front,
            hallway_back,
            meeting_floor,
            meeting_ceiling,
            meeting_left,
            meeting_right,
            meeting_front,
            meeting_back,
        ],
        patch_tags,
    )
    _write_materials_csv(output_dir, which_ertd)


def _write_materials_csv(output_dir: Path, which_ertd: int) -> None:
    """Write the default ERTD broadband materials.csv into `output_dir`.

    Coefficients match the current project setup:
      - hallway absorption = 0.042, scattering = 0.01
      - meeting absorption = 0.12, scattering = 0.01

    Args:
        output_dir (pathlib.Path): Destination folder where `materials.csv`
            will be created.
        hallway_mat (str): Hallway material name used in OBJ/MTL patch tags.
        meeting_mat (str): Meeting-room material name used in OBJ/MTL patch
            tags.

    Returns:
        None
    """

    hallway_right_abs = 0.48 if which_ertd == 2 else 0.042

    materials = {
        "HallwayLeftWall": 0.042,
        "HallwayRightWall": hallway_right_abs,
        "HallwayFrontWall": 0.042,
        "HallwayBackWall": 0.042,
        "HallwayFloor": 0.042,
        "HallwayCeiling": 0.042,
        "MeetingLeftWall": 0.12,
        "MeetingRightWall": 0.12,
        "MeetingFrontWall": 0.12,
        "MeetingBackWall": 0.12,
        "MeetingFloor": 0.12,
        "MeetingCeiling": 0.12,
    }

    lines = ["Frequencies"]

    for name, absorption in materials.items():
        lines.append(f"{name}, {absorption}")
        lines.append(f"{name}, 0.01")

    (output_dir / "materials.csv").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for mesh generation.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed CLI namespace with output path, patch area,
        material names, and RGB colors.
    """
    p = argparse.ArgumentParser(
        description="Generate ERTD mesh.obj/mesh.mtl for a target patch area.")
    p.add_argument("--patch-area",
                   type=float,
                   required=True,
                   help="Target area (m^2) per rectangular patch.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submodules/MoD-ART/environment/ERTD_generated"),
        help="Output folder for mesh.obj and mesh.mtl.",
    )
    p.add_argument("--hallway-material", type=str, default="HallwayWall")
    p.add_argument("--meeting-material", type=str, default="MeetingRoomWall")
    p.add_argument("--hallway-kd",
                   type=float,
                   nargs=3,
                   default=(0.75, 0.62, 0.40))
    p.add_argument("--meeting-kd",
                   type=float,
                   nargs=3,
                   default=(0.35, 0.62, 0.82))
    parser.add_argument("--which-ertd", type=int, default=1)

    return p.parse_args()


def main() -> None:
    """CLI entry point.

    Reads CLI args, builds material styles, generates OBJ/MTL files, and prints
    output paths.

    Args:
        None

    Returns:
        None
    """
    args = _parse_args()

    final_output_dir = Path(
        f"{args.output_dir}_patch_area={args.patch_area:.1f}")
    generate_mesh(final_output_dir, args.patch_area, args.which_ertd)
    print(f"Wrote {final_output_dir / 'mesh.obj'}")
    print(f"Wrote {final_output_dir / 'mesh.mtl'}")


if __name__ == "__main__":
    main()

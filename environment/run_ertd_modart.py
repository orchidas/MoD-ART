import argparse
import os
import sys
from loguru import logger
from pathlib import Path

from raves import raves
from raves.src.utils.raves_io import visualize_mesh


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for mesh generation.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed CLI namespace with output path, patch area,
        material names, and RGB colors.
    """
    p = argparse.ArgumentParser(
        description=
        "Generate ERTD ART/MoD-ART parameters for a target patch area.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submodules/MoD-ART/environment/ERTD_generated"),
        help="Output folder for mesh.obj and mesh.mtl.",
    )
    p.add_argument("--patch-area",
                   type=float,
                   default=None,
                   help="Target area (m^2) per rectangular patch.")
    p.add_argument("--which-ertd",
                   type=int,
                   default=1,
                   help="Which ERTD to simulate, 1 or 2")

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
    if args.patch_area is not None:
        patch_area = args.patch_area
        environment_name = f'ERTD{args.which_ertd}_generated_patch_area={patch_area:.1f}'
        environment_folder = Path(f'environment/{environment_name}')
    else:
        environment_folder = args.output_dir

    if not environment_folder.exists():
        logger.error("Environment does not exist, generate mesh first")
        sys.exit(1)

    image = visualize_mesh(environment_folder.resolve(),
                           interactive_window=False)

    raves(environment_folder.resolve(),
          overwrite=True,
          skip_MoDART=False,
          skip_T60_plots=True)


if __name__ == "__main__":
    main()

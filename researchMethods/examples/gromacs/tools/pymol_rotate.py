#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pymol

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('min', type=float,
                        help="Start point for the torsion")
    parser.add_argument('max', type=float,
                        help="End point for the torsion")
    parser.add_argument('nsteps', type=int,
                        help="Number of torsion steps")
    parser.add_argument('-a', '--atom_idx', nargs=4, type=int, required=True,
                        help="Atom indices for torsion. Example: 6 1 7 8")
    parser.add_argument('-f', '--file', required=True,
                        help="PDB input file")
    parser.add_argument('-n', '--name', required=True,
                        help="PDB file basename")
    parser.add_argument('-d', '--destdir',
                        help="Destination directory")
    return parser.parse_args()

def main():
    args = parse_args()
    angles = np.around(np.linspace(args.min, args.max, args.nsteps), 2)
    if args.destdir is None:
        destdir = Path.cwd()
    else:
        destdir = Path(args.destdir)
        destdir.mkdir(parents=True, exist_ok=True)

    pymol.pymol_argv = ['pymol','-qc']
    pymol.finish_launching()
    cmd = pymol.cmd

    cmd.load(args.file)
    cmd.set("retain_order", '1')

    for angle in angles:
        outfile = destdir.joinpath(f'{args.name}_{angle:07.2f}.pdb')
        cmd.set_dihedral(f'id {args.atom_idx[0]}',
                         f'id {args.atom_idx[1]}',
                         f'id {args.atom_idx[2]}',
                         f'id {args.atom_idx[3]}',
                         angle)
        cmd.save(str(outfile))

if __name__ == '__main__':
    main()

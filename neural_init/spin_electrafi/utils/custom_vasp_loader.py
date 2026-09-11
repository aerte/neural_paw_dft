import re
import os
import time
import ase.io.vasp as aiv
import numpy as np
import ase
from ase.calculators.vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
import torch
import lz4.frame
import re
import os
import numpy as np
import ase
from ase.calculators.vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator


def get_vasp_version(string):
    """Extract version number from header of stdout.

    Example::

      >>> get_vasp_version('potato vasp.6.1.2 bumblebee')
      '6.1.2'

    """
    match = re.search(r'vasp\.(\S+)', string, re.M)
    return match.group(1)


class CustomVaspChargeDensity:
    """Class for representing VASP charge density.

    Filename is normally CHG."""
    # Can the filename be CHGCAR?  There's a povray tutorial
    # in doc/tutorials where it's CHGCAR as of January 2021.  --askhl
    def __init__(self, filename):
        # Instance variables
        self.atoms = []  # List of Atoms objects
        self.chg = []  # Charge density
        self.chgdiff = []  # Charge density difference, if spin polarized
        self.aug = ''  # Augmentation charges, not parsed just a big string
        self.augdiff = ''  # Augmentation charge differece, is spin polarized

        # Note that the augmentation charge is not a list, since they
        # are needed only for CHGCAR files which store only a single
        # image.
        if filename is not None:
            self.read(filename)

    def is_spin_polarized(self):
        if len(self.chgdiff) > 0:
            return True
        return False

    def _read_chg(self, fobj, chg, volume):
        """Read charge from file object

        Utility method for reading the actual charge density (or
        charge density difference) from a file object. On input, the
        file object must be at the beginning of the charge block, on
        output the file position will be left at the end of the
        block. The chg array must be of the correct dimensions.

        """
        # VASP writes charge density as
        # WRITE(IU,FORM) (((C(NX,NY,NZ),NX=1,NGXC),NY=1,NGYZ),NZ=1,NGZC)
        # Fortran nested implied do loops; innermost index fastest
        # First, just read it in
        for zz in range(chg.shape[2]):
            for yy in range(chg.shape[1]):
                chg[:, yy, zz] = np.fromfile(fobj, count=chg.shape[0], sep=' ')
        chg /= volume

    def read(self, filename):
        """
        Robust CHG/CHGCAR reader:
        - Reads sequentially from a single file handle (no reopen).
        - Handles spin blocks whose header line also contains the first data values.
        - Populates self.atoms (list), self.chg (list of 3D arrays),
          and self.chgdiff (list) if a spin-difference grid is present.
        """
        import ase.io.vasp as aiv
        import numpy as np

        def _read_dims_and_leading_values(fd):
            # Read next non-empty line and parse dims + any trailing floats
            line = fd.readline()
            while line and line.strip() == "":
                line = fd.readline()
            if not line:
                return None, None  # EOF

            toks = line.split()
            if len(toks) < 3:
                raise RuntimeError("Malformed CHGCAR: missing nx ny nz line.")
            try:
                nx, ny, nz = map(int, toks[:3])
            except Exception as e:
                # Not a dims line at all (caller will handle by rewinding)
                raise

            # Any extra tokens on the same line are the first data values
            lead = []
            for t in toks[3:]:
                try:
                    lead.append(float(t))
                except ValueError:
                    # stop if we hit non-numeric (e.g., "augmentation")
                    break
            return (nx, ny, nz), np.asarray(lead, dtype=float)

        def _read_grid(fd, nx, ny, nz, lead_vals, volume):
            N = nx * ny * nz
            k = 0 if lead_vals is None else int(lead_vals.size)
            need = N - k
            if need < 0:
                raise RuntimeError(f"Header line overfull: {k} > N={N}")

            arr_rem = np.empty(0, dtype=float)
            if need > 0:
                # np.fromfile with sep=' ' reads across line breaks
                arr_rem = np.fromfile(fd, count=need, sep=' ')
                if arr_rem.size != need:
                    raise ValueError(
                        f"Unexpected end of grid: needed {N}, got {k + arr_rem.size}"
                    )

            full = arr_rem if k == 0 else np.concatenate([lead_vals, arr_rem])
            # VASP writes Fortran order: x fastest, then y, then z
            grid = np.reshape(full, (nx, ny, nz), order='F') / volume
            return grid

        self.atoms = []
        self.chg = []
        self.chgdiff = []  # make it a list (never None)
        self.aug = ""
        self.augdiff = ""

        with open(filename, "r") as fd:
            while True:
                # Try to read one POSCAR header (one “image”)
                try:
                    atoms = aiv.read_vasp_configuration(fd)
                except (IOError, ValueError, IndexError, RuntimeError):
                    break  # no more images / malformed start

                # There is usually one blank line after POSCAR
                _ = fd.readline()

                vol = atoms.get_volume()

                # --- total charge block ---
                dims, lead = _read_dims_and_leading_values(fd)
                if dims is None:
                    break
                nx, ny, nz = dims
                total = _read_grid(fd, nx, ny, nz, lead, vol)
                self.chg.append(total)
                self.atoms.append(atoms)

                # --- peek for second grid (spin-difference) ---
                pos = fd.tell()
                line = fd.readline()
                while line and line.strip() == "":
                    pos = fd.tell()
                    line = fd.readline()
                if not line:
                    break  # EOF after total grid

                tokens = line.split()
                looks_like_dims = False
                if len(tokens) >= 3:
                    t0, t1, t2 = tokens[:3]

                    def _is_int(tok):
                        s = tok.strip()
                        if s.startswith(("+", "-")):
                            s = s[1:]
                        return s.isdigit()

                    looks_like_dims = _is_int(t0) and _is_int(t1) and _is_int(t2)

                if looks_like_dims:
                    # parse dims + possible inline values from the line we already read
                    try:
                        nx2, ny2, nz2 = map(int, tokens[:3])
                    except Exception:
                        fd.seek(pos)  # not a true dims line; rewind
                    else:
                        lead2 = []
                        for t in tokens[3:]:
                            try:
                                lead2.append(float(t))
                            except ValueError:
                                break
                        lead2 = np.asarray(lead2, dtype=float)
                        # Read remaining spin-diff values
                        spin = _read_grid(fd, nx2, ny2, nz2, lead2, vol)
                        self.chgdiff.append(spin)
                else:
                    # Not a dims line; rewind and leave for augmentation / next image
                    fd.seek(pos)

                # Optionally capture augmentation blocks if you rely on them for writing
                # Consume until the next POSCAR or EOF if needed.
                # (Most workflows don’t need aug/augdiff and can leave them empty.)

        # If nothing was read, keep types consistent
        if len(self.chgdiff) == 0:
            self.chgdiff = []  # explicit empty list

    def _write_chg(self, fobj, chg, volume, format='chg'):
        """Write charge density

        Utility function similar to _read_chg but for writing.

        """
        # Make a 1D copy of chg, must take transpose to get ordering right
        chgtmp = chg.T.ravel()
        # Multiply by volume
        chgtmp = chgtmp * volume
        # Must be a tuple to pass to string conversion
        chgtmp = tuple(chgtmp)
        # CHG format - 10 columns
        if format.lower() == 'chg':
            # Write all but the last row
            for ii in range((len(chgtmp) - 1) // 10):
                fobj.write(' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G\
 %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G\n' % chgtmp[ii * 10:(ii + 1) * 10])
            # If the last row contains 10 values then write them without a
            # newline
            if len(chgtmp) % 10 == 0:
                fobj.write(' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G'
                           ' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G' %
                           chgtmp[len(chgtmp) - 10:len(chgtmp)])
            # Otherwise write fewer columns without a newline
            else:
                for ii in range(len(chgtmp) % 10):
                    fobj.write((' %#11.5G') %
                               chgtmp[len(chgtmp) - len(chgtmp) % 10 + ii])
        # Other formats - 5 columns
        else:
            # Write all but the last row
            for ii in range((len(chgtmp) - 1) // 5):
                fobj.write(' %17.10E %17.10E %17.10E %17.10E %17.10E\n' %
                           chgtmp[ii * 5:(ii + 1) * 5])
            # If the last row contains 5 values then write them without a
            # newline
            if len(chgtmp) % 5 == 0:
                fobj.write(' %17.10E %17.10E %17.10E %17.10E %17.10E' %
                           chgtmp[len(chgtmp) - 5:len(chgtmp)])
            # Otherwise write fewer columns without a newline
            else:
                for ii in range(len(chgtmp) % 5):
                    fobj.write((' %17.10E') %
                               chgtmp[len(chgtmp) - len(chgtmp) % 5 + ii])
        # Write a newline whatever format it is
        fobj.write('\n')

    def write(self, filename, format=None):
        """Write VASP charge density in CHG format.

        filename: str
            Name of file to write to.
        format: str
            String specifying whether to write in CHGCAR or CHG
            format.

        """
        import ase.io.vasp as aiv
        if format is None:
            if filename.lower().find('chgcar') != -1:
                format = 'chgcar'
            elif filename.lower().find('chg') != -1:
                format = 'chg'
            elif len(self.chg) == 1:
                format = 'chgcar'
            else:
                format = 'chg'
        with open(filename, 'w') as fd:
            for ii, chg in enumerate(self.chg):
                if format == 'chgcar' and ii != len(self.chg) - 1:
                    continue  # Write only the last image for CHGCAR
                aiv.write_vasp(fd,
                               self.atoms[ii],
                               direct=True,
                               #long_format=False
                               )
                fd.write('\n')
                for dim in chg.shape:
                    fd.write(' %4i' % dim)
                fd.write('\n')
                vol = self.atoms[ii].get_volume()
                self._write_chg(fd, chg, vol, format)
                if format == 'chgcar':
                    fd.write(self.aug)
                if self.is_spin_polarized():
                    if format == 'chg':
                        fd.write('\n')
                    for dim in chg.shape:
                        fd.write(' %4i' % dim)
                    fd.write('\n')  # a new line after dim is required
                    self._write_chg(fd, self.chgdiff[ii], vol, format)
                    if format == 'chgcar':
                        # a new line is always provided self._write_chg
                        fd.write(self.augdiff)
                if format == 'chg' and len(self.chg) > 1:
                    fd.write('\n')

    def READ_CHG_CUSTOM(self, fobj, chg, atoms):
        """
        Read one total charge grid; if a second (nx ny nz) header follows,
        read a spin-difference grid. Handles the case where header and the
        first data values share the same line: "nx ny nz v1 v2 ...".
        """
        import numpy as np

        vol = atoms.get_volume()
        nx, ny, nz = map(int, chg.shape)
        N = nx * ny * nz

        # --- read total charge: exactly N numbers (Fortran order) ---
        arr = np.fromfile(fobj, count=N, sep=' ')
        if arr.size != N:
            raise ValueError(f"Unexpected end of CHGCAR: needed {N}, got {arr.size}")
        chg[:] = np.reshape(arr, (nx, ny, nz), order='F') / vol

        # --- peek next non-empty line ---
        pos = fobj.tell()
        line = fobj.readline()
        while line is not None and line.strip() == '':
            pos = fobj.tell()
            line = fobj.readline()
        if not line:
            return chg  # EOF

        tokens = line.split()

        # header if first 3 tokens are integers (allow + / - signs)
        def _is_int(tok: str) -> bool:
            s = tok.strip()
            if s.startswith(('+', '-')): s = s[1:]
            return s.isdigit()

        looks_like_dims = len(tokens) >= 3 and all(_is_int(t) for t in tokens[:3])

        if looks_like_dims:
            nx2, ny2, nz2 = map(int, tokens[:3])
            N2 = nx2 * ny2 * nz2

            # Any *extra tokens on the same line* are the first spin values
            lead_vals = []
            if len(tokens) > 3:
                # Try to parse floats like '-.48680E-01'
                for t in tokens[3:]:
                    try:
                        lead_vals.append(float(t))
                    except ValueError:
                        # if a non-numeric sneaks in (e.g., 'augmentation'), stop
                        break

            k = len(lead_vals)
            need = N2 - k
            if need < 0:
                raise ValueError(f"Spin grid overfull on header line: have {k} > N={N2}")

            arr2 = np.empty(0, dtype=float)
            if need > 0:
                arr2 = np.fromfile(fobj, count=need, sep=' ')
                if arr2.size != need:
                    raise ValueError(
                        f"Unexpected end of CHGCAR spin grid: needed {N2}, got {k + arr2.size}"
                    )

            full = np.concatenate([np.asarray(lead_vals, dtype=float), arr2])
            chgdiff = np.reshape(full, (nx2, ny2, nz2), order='F') / vol

            if not hasattr(self, 'chgdiff') or self.chgdiff is None:
                self.chgdiff = []
            self.chgdiff.append(chgdiff)
        else:
            # Not a second grid; rewind to let caller process augmentation text
            fobj.seek(pos)

        return chg


class VaspDos:
    """Class for representing density-of-states produced by VASP

    The energies are in property self.energy

    Site-projected DOS is accesible via the self.site_dos method.

    Total and integrated DOS is accessible as numpy.ndarray's in the
    properties self.dos and self.integrated_dos. If the calculation is
    spin polarized, the arrays will be of shape (2, NDOS), else (1,
    NDOS).

    The self.efermi property contains the currently set Fermi
    level. Changing this value shifts the energies.

    """
    def __init__(self, doscar='DOSCAR', efermi=0.0):
        """Initialize"""
        self._efermi = 0.0
        self.read_doscar(doscar)
        self.efermi = efermi

        # we have determine the resort to correctly map ase atom index to the
        # POSCAR.
        self.sort = []
        self.resort = []
        if os.path.isfile('ase-sort.dat'):
            file = open('ase-sort.dat', 'r')
            lines = file.readlines()
            file.close()
            for line in lines:
                data = line.split()
                self.sort.append(int(data[0]))
                self.resort.append(int(data[1]))

    def _set_efermi(self, efermi):
        """Set the Fermi level."""
        ef = efermi - self._efermi
        self._efermi = efermi
        self._total_dos[0, :] = self._total_dos[0, :] - ef
        try:
            self._site_dos[:, 0, :] = self._site_dos[:, 0, :] - ef
        except IndexError:
            pass

    def _get_efermi(self):
        return self._efermi

    efermi = property(_get_efermi, _set_efermi, None, "Fermi energy.")

    def _get_energy(self):
        """Return the array with the energies."""
        return self._total_dos[0, :]

    energy = property(_get_energy, None, None, "Array of energies")

    def site_dos(self, atom, orbital):
        """Return an NDOSx1 array with dos for the chosen atom and orbital.

        atom: int
            Atom index
        orbital: int or str
            Which orbital to plot

        If the orbital is given as an integer:
        If spin-unpolarized calculation, no phase factors:
        s = 0, p = 1, d = 2
        Spin-polarized, no phase factors:
        s-up = 0, s-down = 1, p-up = 2, p-down = 3, d-up = 4, d-down = 5
        If phase factors have been calculated, orbitals are
        s, py, pz, px, dxy, dyz, dz2, dxz, dx2
        double in the above fashion if spin polarized.

        """
        # Correct atom index for resorting if we need to. This happens when the
        # ase-sort.dat file exists, and self.resort is not empty.
        if self.resort:
            atom = self.resort[atom]

        # Integer indexing for orbitals starts from 1 in the _site_dos array
        # since the 0th column contains the energies
        if isinstance(orbital, int):
            return self._site_dos[atom, orbital + 1, :]
        n = self._site_dos.shape[1]

        from ase.calculators.vasp.vasp_data import PDOS_orbital_names_and_DOSCAR_column
        norb = PDOS_orbital_names_and_DOSCAR_column[n]

        return self._site_dos[atom, norb[orbital.lower()], :]

    def _get_dos(self):
        if self._total_dos.shape[0] == 3:
            return self._total_dos[1, :]
        elif self._total_dos.shape[0] == 5:
            return self._total_dos[1:3, :]

    dos = property(_get_dos, None, None, 'Average DOS in cell')

    def _get_integrated_dos(self):
        if self._total_dos.shape[0] == 3:
            return self._total_dos[2, :]
        elif self._total_dos.shape[0] == 5:
            return self._total_dos[3:5, :]

    integrated_dos = property(_get_integrated_dos, None, None,
                              'Integrated average DOS in cell')

    def read_doscar(self, fname="DOSCAR"):
        """Read a VASP DOSCAR file"""
        fd = open(fname)
        natoms = int(fd.readline().split()[0])
        [fd.readline() for nn in range(4)]  # Skip next 4 lines.
        # First we have a block with total and total integrated DOS
        ndos = int(fd.readline().split()[2])
        dos = []
        for nd in range(ndos):
            dos.append(np.array([float(x) for x in fd.readline().split()]))
        self._total_dos = np.array(dos).T
        # Next we have one block per atom, if INCAR contains the stuff
        # necessary for generating site-projected DOS
        dos = []
        for na in range(natoms):
            line = fd.readline()
            if line == '':
                # No site-projected DOS
                break
            ndos = int(line.split()[2])
            line = fd.readline().split()
            cdos = np.empty((ndos, len(line)))
            cdos[0] = np.array(line)
            for nd in range(1, ndos):
                line = fd.readline().split()
                cdos[nd] = np.array([float(x) for x in line])
            dos.append(cdos.T)
        self._site_dos = np.array(dos)
        fd.close()


class xdat2traj:
    def __init__(self,
                 trajectory=None,
                 atoms=None,
                 poscar=None,
                 xdatcar=None,
                 sort=None,
                 calc=None):
        """
        trajectory is the name of the file to write the trajectory to
        poscar is the name of the poscar file to read. Default: POSCAR
        """
        if not poscar:
            self.poscar = 'POSCAR'
        else:
            self.poscar = poscar

        if not atoms:
            # This reads the atoms sorted the way VASP wants
            self.atoms = ase.io.read(self.poscar, format='vasp')
            resort_reqd = True
        else:
            # Assume if we pass atoms that it is sorted the way we want
            self.atoms = atoms
            resort_reqd = False

        if not calc:
            self.calc = Vasp()
        else:
            self.calc = calc
        if not sort:
            if not hasattr(self.calc, 'sort'):
                self.calc.sort = list(range(len(self.atoms)))
        else:
            self.calc.sort = sort
        self.calc.resort = list(range(len(self.calc.sort)))
        for n in range(len(self.calc.resort)):
            self.calc.resort[self.calc.sort[n]] = n

        if not xdatcar:
            self.xdatcar = 'XDATCAR'
        else:
            self.xdatcar = xdatcar

        if not trajectory:
            self.trajectory = 'out.traj'
        else:
            self.trajectory = trajectory

        self.out = ase.io.trajectory.Trajectory(self.trajectory, mode='w')

        if resort_reqd:
            self.atoms = self.atoms[self.calc.resort]
        self.energies = self.calc.read_energy(all=True)[1]
        # Forces are read with the atoms sorted using resort
        self.forces = self.calc.read_forces(self.atoms, all=True)

    def convert(self):
        lines = open(self.xdatcar).readlines()
        if len(lines[7].split()) == 0:
            del (lines[0:8])
        elif len(lines[5].split()) == 0:
            del (lines[0:6])
        elif len(lines[4].split()) == 0:
            del (lines[0:5])
        elif lines[7].split()[0] == 'Direct':
            del (lines[0:8])
        step = 0
        iatom = 0
        scaled_pos = []
        for line in lines:
            if iatom == len(self.atoms):
                if step == 0:
                    self.out.write_header(self.atoms[self.calc.resort])
                scaled_pos = np.array(scaled_pos)
                # Now resort the positions to match self.atoms
                self.atoms.set_scaled_positions(scaled_pos[self.calc.resort])

                calc = SinglePointCalculator(self.atoms,
                                             energy=self.energies[step],
                                             forces=self.forces[step])
                self.atoms.calc = calc
                self.out.write(self.atoms)
                scaled_pos = []
                iatom = 0
                step += 1
            else:
                if not line.split()[0] == 'Direct':
                    iatom += 1
                    scaled_pos.append(
                        [float(line.split()[n]) for n in range(3)])

        # Write also the last image
        # I'm sure there is also more clever fix...
        if step == 0:
            self.out.write_header(self.atoms[self.calc.resort])
        scaled_pos = np.array(scaled_pos)[self.calc.resort]
        self.atoms.set_scaled_positions(scaled_pos)
        calc = SinglePointCalculator(self.atoms,
                                     energy=self.energies[step],
                                     forces=self.forces[step])
        self.atoms.calc = calc
        self.out.write(self.atoms)

        self.out.close()


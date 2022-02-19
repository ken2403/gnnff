import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as md
import numpy as np
import torch
import ase
import ase.io

import gnnff

__all__ = ["FromPoscarToXml"]

FILE_NAME_RUN = "gnnff_run.xml"


class FromPoscarToXml:
    """
    From POSCAR structure file and learned SchNetTriple model,
    calculate total energy and atomic forces.
    Output file is .xml, because it is useful for phonon calculation by phonopy.

    Parameters
    ----------
    poscar_path: str
        path to POSCAR file
    model_path: str
        path to learned model
    cuda: bool
        if True, computing on GPU
    """

    def __init__(self, poscar_path: str, model_path: str, cuda: bool) -> None:
        self.poscar_path = poscar_path
        self.model_path = model_path
        self.device = torch.device("cuda" if cuda else "cpu")

    def __call__(
        self,
        cutoff: float,
    ) -> None:
        """
        From POSCAR structure file and learned SchNetTriple model,
        calculate total energy and atomic forces, and output them to .xml file,
        because it is useful for phonon calculation using phonopy.

        Parameters
        ----------
        cutoff : float, optional
            cutoff radious
        """
        inputs, at = self.from_poscar(cutoff=cutoff)
        self.to_xml(inputs=inputs, atoms=at)

    def from_poscar(
        self,
        cutoff: float,
    ):
        """
        Parameters
        ----------
        cutoff : float, optional
            cutoff radious

        Returns
        -------
        inputs : dict of torch.Tensor
            Properties including neighbor lists and masks reformated into SchNetTriple input format.
        at : ase.Atoms
            atoms object of POSCAR structure.
        """
        # get atoms object from POSCAR
        at = ase.io.read(self.poscar_path, format="vasp")
        # construct converter
        converter = gnnff.data.celldata.CellDataConverter(
            cutoff=cutoff, device=self.device
        )
        # convert atoms object
        inputs = converter(at)

        return inputs, at

    def _convert_xml(pred, atoms):
        # get information of structure
        positions = atoms.get_positions()
        basis = np.array(atoms.get_cell())
        volume = np.array([atoms.get_volume()])

        # get information of prediction values
        forces = pred["forces"].detach().cpu().numpy()[0]

        # make xml tree like vasprun.xml
        root = ET.Element("modeling")

        generator = ET.SubElement(root, "generator")
        i1 = ET.SubElement(generator, "i")
        i1.set("name", "program")
        i1.set("type", "string")
        i1.text = "schnet"
        i2 = ET.SubElement(generator, "i")
        i2.set("name", "version")
        i2.set("type", "string")
        i2.text = "v1.0.0"

        calculation = ET.SubElement(root, "calculation")
        structure = ET.SubElement(calculation, "structure")

        crystal = ET.SubElement(structure, "crystal")
        varray1 = ET.SubElement(crystal, "varray")
        varray1.set("name", "basis")
        i3 = ET.SubElement(crystal, "i")
        i3.set("name", "volume")

        varray2 = ET.SubElement(structure, "varray")
        varray2.set("name", "positions")

        varray3 = ET.SubElement(calculation, "varray")
        varray3.set("name", "forces")

        # add infomation to tree
        for base in basis:
            v = ET.SubElement(varray1, "v")
            v.text = f"\t{base[0]:>11.8f}\t{base[1]:>11.8f}\t{base[2]:>11.8f} "

        i3.text = f"\t{volume[0]:>11.8f} "

        for pos in positions:
            v = ET.SubElement(varray2, "v")
            v.text = f"\t{pos[0]:>11.8f}\t{pos[1]:>11.8f}\t{pos[2]:>11.8f} "

        for force in forces:
            v = ET.SubElement(varray3, "v")
            v.text = f"\t{force[0]:>11.8f}\t{force[1]:>11.8f}\t{force[2]:>11.8f} "

        return root

    def to_xml(
        self,
        inputs: dict,
        atoms: ase.Atoms,
        save_path: str = None,
    ):
        """

        Parameters
        ----------
        inputs : dict of torch.Tensor
            Properties including neighbor lists and masks reformated into SchNetTriple input format.
        atoms : ase.Atoms
            atoms object of structure.
        save_path : str, default=None
            save path of result .xml file. Default is same dirctory as POSCAR file.
        """
        if save_path is None:
            save_path = os.path.pardir(os.path.abspath(self.poscar_path))
        save_path = os.path.join(save_path, FILE_NAME_RUN)

        model = torch.load(self.model_path, map_location=self.device)
        pred = model(inputs)

        xml = self._convert_xml(pred, atoms)
        # 文字列パースを介してminidomへ移す
        document = md.parseString(ET.tostring(xml, "utf-8"))
        file = open(save_path, "w")
        # エンコーディング、改行、全体のインデント、子要素の追加インデントを設定しつつファイルへ書き出し
        document.writexml(file, encoding="utf-8", newl="\n", indent="", addindent="  ")
        file.close()

import base64
import functools
import rdkit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib_inline
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Draw
from rdkit.Chem import DataStructs
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D

def red_blue_cmap(x):
    """Red to Blue color map
    Args:
        x (float): value between -1 ~ 1, represents normalized saliency score
    Returns (tuple): tuple of 3 float values representing R, G, B.
    """
    if x > 0:
        # Red for positive value
        # x=0 -> 1, 1, 1  (white)
        # x=1 -> 1, 0, 0 (red)
        return 1.0, 1.0-x, 1.0-x
    else:
        # Blue for negative value
        x *= -1
        return 1.0-x, 1.0-x, 1.0

def is_visible(begin, end):
    if begin <= 0 or end <= 0:
        return 0
    elif begin >= 1 or end >= 1:
        return 1
    else:
        return (begin+end) * 0.5
def color_bond(bond, saliency, color_fn):
    begin = saliency[bond.GetBeginAtomIdx()]
    end = saliency[bond.GetEndAtomIdx()]
    return color_fn(is_visible(begin, end))

def label_cat(label):
    return '$+$' if bool(label!=0) else '$\cdot$'

class Mol2Img():
    def __init__(self, mol, atom_colors, bond_colors, molSize=(450, 150), kekulize=True):
        self.mol = mol
        self.atom_colors = atom_colors
        self.bond_colors = bond_colors
        self.molSize = molSize
        self.kekulize = kekulize
        self.mc = Chem.Mol(self.mol.ToBinary())
        if self.kekulize:
            try:
                Chem.Kekulize(self.mc)
            except:
                self.mc = Chem.Mol(self.mol.ToBinary())

    def mol2png(self):
        drawer = rdMolDraw2D.MolDraw2DCairo(self.molSize[0], self.molSize[1])
        self._getDrawingText(drawer)
        return drawer.GetDrawingText()
    
    def mol2svg(self):
        drawer = rdMolDraw2D.MolDraw2DSVG(self.molSize[0], self.molSize[1])
        self._getDrawingText(drawer)
        return drawer.GetDrawingText()
    
    def _getDrawingText(self, drawer):
            dops = drawer.drawOptions()
            dops.useBWAtomPalette()
            dops.padding = .2
            dops.addAtomIndices = True
            drawer.DrawMolecule(
                self.mc,
                highlightAtoms=[i for i in range(len(self.atom_colors))], 
                highlightAtomColors=self.atom_colors, 
                highlightBonds=[i for i in range(len(self.bond_colors))],
                highlightBondColors=self.bond_colors,
                highlightAtomRadii={i: .5 for i in range(len(self.atom_colors))}
                )
            drawer.FinishDrawing()
            
class XMol():
    def __init__(self, mol, weight_fn, weights=None, atoms=['C', 'N', 'O', 'S', 'F', 'Cl', 'P', 'Br'], drawingfmt='svg'):
        self.mol = mol
        self.weight_fn = weight_fn
        self.weights = weights
        self.atoms = atoms
        self.drawingfmt = drawingfmt

    def make_explainable_image(self, kekulize=True, molSize=(450, 150)):
        symbols = [f'{self.mol.GetAtomWithIdx(i).GetSymbol()}_{i}' for i in range(self.mol.GetNumAtoms())]
        #df = pd.DataFrame(columns=self.atoms)
        if self.weights is None:
            contribs = self.weight_fn(self.mol)
        else:
            contribs = self.weights
        num_atoms = self.mol.GetNumAtoms()
        arr = np.zeros((num_atoms, len(self.atoms)))
        for i in range(self.mol.GetNumAtoms()):
            _a = self.mol.GetAtomWithIdx(i).GetSymbol()
            arr[i, self.atoms.index(_a)] = contribs[i]
        df = pd.DataFrame(arr, index=symbols, columns=self.atoms)
        self.weights, self.vmax = SimilarityMaps.GetStandardizedWeights(contribs)
        self.vmin = -self.vmax
        atom_colors = {i: red_blue_cmap(e) for i, e in enumerate(self.weights)}
        # bondlist = [bond.GetIdx() for bond in mol.GetBonds()]
        bond_colors = {i: color_bond(bond, self.weights, red_blue_cmap) for i, bond in enumerate(self.mol.GetBonds())}
        viz = Mol2Img(self.mol, atom_colors, bond_colors, molSize=molSize, kekulize=kekulize)
        if self.drawingfmt == 'svg':
            matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
            self.drawingtxt = viz.mol2svg()
        elif self.drawingfmt == 'png':
            self.drawingtext = viz.mol2png()
            matplotlib_inline.backend_inline.set_matplotlib_formats('png')
        else:
            raise Exception("Please select drawingfmt form 'svg' or 'png'")
        self.fig = plt.figure(figsize=(18, 9))
        self.grid = plt.GridSpec(15, 10)
        self.ax = self.fig.add_subplot(self.grid[1:, -1])
        self.ax.barh(range(self.mol.GetNumAtoms()), np.maximum(0, df.values).sum(axis=1), color='C3')
        self.ax.barh(range(self.mol.GetNumAtoms()), np.minimum(0, df.values).sum(axis=1), color='C0')
        self.ax.set_yticks(range(self.mol.GetNumAtoms()))
        self.ax.set_ylim(-.5, self.mol.GetNumAtoms()-0.5)
        symbols= {i: f'${self.mol.GetAtomWithIdx(i).GetSymbol()}_{{{i}}}$' for i in range(self.mol.GetNumAtoms())}
        self.ax.axvline(0, color='k', linestyle='-', linewidth=.5)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.tick_params(axis='both', which='both', left=False, labelleft=False)

        self.ax = self.fig.add_subplot(self.grid[1:, :-1], sharey=self.ax)
        self.im = self.ax.imshow(df.values, cmap='bwr', vmin=self.vmin, vmax=self.vmax, aspect='auto')
        self.ax.set_yticks(range(self.mol.GetNumAtoms()))
        self.ax.set_ylim(self.mol.GetNumAtoms() -.5, -.5)
        self.symbols= {i: f'${self.mol.GetAtomWithIdx(i).GetSymbol()}_{{{i}}}$' for i in range(self.mol.GetNumAtoms())}
        self.ax.set_yticklabels(symbols.values())
        self.ax.set_xticks(range(len(self.atoms)))


        self.ax.set_xlim(-.5, len(self.atoms) -.5)
        self.ax.set_xticklabels(self.atoms, rotation=90)
        self.ax.set_ylabel('Node')

        for (j,i),label in np.ndenumerate(df.values):
            self.ax.text(i,j, label_cat(label) ,ha='center',va='center')
        self.ax.tick_params(axis='both', which='both', bottom=True, labelbottom=True, top=False, labeltop=False)
        #ax.grid(c=None)

        self.ax = self.fig.add_subplot(self.grid[0, :-1])
        self.fig.colorbar(mappable=self.im, cax=self.ax, orientation='horizontal')
        self.ax.tick_params(axis='both', which='both', bottom=False, labelbottom=False, top=True, labeltop=True)
        
import marimo

__generated_with = "0.2.3"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import chemviz
    return chemviz, mo


@app.cell
def __():
    import functools
    import os
    import warnings
    warnings.filterwarnings(action='ignore')

    from matplotlib import cm
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import rdkit
    from rdkit import Chem, RDPaths
    from rdkit.Chem import AllChem,  DataStructs, Draw, rdBase, rdCoordGen, rdDepictor
    from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D, SimilarityMaps
    from rdkit.ML.Descriptors import MoleculeDescriptors
    print(f'RDKit: {rdBase.rdkitVersion}')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import shap
    print(f'SHAP: {shap.__version__}')
    return (
        AllChem,
        Chem,
        DataStructs,
        Draw,
        IPythonConsole,
        MoleculeDescriptors,
        RDPaths,
        RandomForestClassifier,
        SimilarityMaps,
        accuracy_score,
        cm,
        functools,
        gridspec,
        np,
        os,
        pd,
        plt,
        rdBase,
        rdCoordGen,
        rdDepictor,
        rdMolDraw2D,
        rdkit,
        shap,
        warnings,
    )


@app.cell
def __(
    AllChem,
    Chem,
    DataStructs,
    RDPaths,
    RandomForestClassifier,
    accuracy_score,
    np,
    os,
):
    def mol2fp(mol,radius=2, nBits=1024):
        bitInfo={}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bitInfo)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr, bitInfo

    train_path = os.path.join(RDPaths.RDDocsDir, 'Book/data/solubility.train.sdf')
    test_path = os.path.join(RDPaths.RDDocsDir, 'Book/data/solubility.test.sdf')
    train_path='/home/iwatobipen/miniforge3/pkgs/rdkit-2023.09.3-py311h4c2f14b_1/share/RDKit/Docs/Book/data/solubility.train.sdf'
    test_path='/home/iwatobipen/miniforge3/pkgs/rdkit-2023.09.3-py311h4c2f14b_1/share/RDKit/Docs/Book/data/solubility.test.sdf'
    train_mols = Chem.SDMolSupplier(train_path)
    test_mols = Chem.SDMolSupplier(test_path)
    print(len(train_mols), len(test_mols))

    sol_classes = {'(A) low': 0, '(B) medium': 1, '(C) high': 2}
    X_train = np.array([mol2fp(m)[0] for m in train_mols])
    y_train = np.array([sol_classes[m.GetProp('SOL_classification')] for m in train_mols], dtype=np.int_) 
    X_test = np.array([mol2fp(m)[0] for m in test_mols])
    y_test = np.array([sol_classes[m.GetProp('SOL_classification')] for m in test_mols], dtype=np.int_)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    clf = RandomForestClassifier(random_state=20191215)
    clf.fit(X_train, y_train)
    print(accuracy_score(y_test, clf.predict(X_test)))
    return (
        X_test,
        X_train,
        clf,
        mol2fp,
        sol_classes,
        test_mols,
        test_path,
        train_mols,
        train_path,
        y_test,
        y_train,
    )


@app.cell
def __(SimilarityMaps, functools, mol2fp, sol_classes):
    def get_proba(fp, proba_fn, class_id):
        return proba_fn((fp,))[0][class_id]

    def fp_partial(nBits):
        return functools.partial(SimilarityMaps.GetMorganFingerprint, nBits=nBits)

    def show_pred_results(mol, model):
        y_pred = model.predict(mol2fp(mol)[0].reshape((1,-1)))
        sol_dict = {val: key for key, val in sol_classes.items()}
        print(f"True: {mol.GetProp('SOL_classification')} vs Predicted: {sol_dict[y_pred[0]]}")
    return fp_partial, get_proba, show_pred_results


@app.cell
def __(Chem, SimilarityMaps, chemviz, clf, fp_partial, get_proba):
    def makeimg(smi):
        mol = Chem.MolFromSmiles(smi)
        weights = SimilarityMaps.GetAtomicWeightsForModel(mol, 
                                                          fp_partial(1024), 
                                                          lambda x:get_proba(x, clf.predict_proba, 2))
        xmol=chemviz.XMol(mol, weight_fn=None, weights=weights)
        xmol.make_explainable_image()
        return xmol.drawingtxt
    return makeimg,


@app.cell
def __(mo):
    form = mo.ui.text(label="input SMILES")
    form
    return form,


@app.cell
def __(form, makeimg, mo):
    mo.Html(f""""{makeimg(form.value)}""")
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

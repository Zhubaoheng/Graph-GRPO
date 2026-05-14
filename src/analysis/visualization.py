import logging
import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
import rdkit.Chem
import swanlab

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    logger.warning("[Visualization] Matplotlib import failed (%s). Plotting will be disabled.", e)


def _ensure_matplotlib():
    if plt is None:
        raise RuntimeError(
            "Matplotlib is unavailable in the current environment, so plotting features are disabled."
        )




class MolecularVisualization:
    def __init__(self, remove_h, dataset_infos):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            logger.info("Can't kekulize molecule")
            mol = None
        return mol

    def visualize(
        self, path: str, molecules: list, num_molecules_to_visualize: int, log="graph", filenames=None
    ):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        if num_molecules_to_visualize > len(molecules):
            logger.info("Shortening to %s", len(molecules))
            num_molecules_to_visualize = len(molecules)

        for i in range(num_molecules_to_visualize):
            if filenames and i < len(filenames):
                file_name = filenames[i]
                if not file_name.endswith(".png"):
                    file_name += ".png"
                file_path = os.path.join(path, file_name)
            else:
                file_path = os.path.join(path, "molecule_{}.png".format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            try:
                Draw.MolToFile(mol, file_path)
                try:
                    if swanlab.run and log is not None:
                        logger.info("Saving %s to swanlab", file_path)
                        # SwanLab uses PIL Image or file path directly
                        from PIL import Image
                        img = Image.open(file_path)
                        swanlab.log({log: swanlab.Image(img)})
                except Exception:
                    pass
            except rdkit.Chem.KekulizeException:
                logger.info("Can't kekulize molecule")

    def visualize_chain(self, path, nodes_list, adjacency_matrix, times, trainer=None):
        RDLogger.DisableLog("rdApp.*")
        # convert graphs to the rdkit molecules
        mols = [
            self.mol_from_graphs(nodes_list[i], adjacency_matrix[i])
            for i in range(nodes_list.shape[0])
        ]

        # find the coordinates of atoms in the final molecule
        final_molecule = mols[-1]
        AllChem.Compute2DCoords(final_molecule)

        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append((positions.x, positions.y, positions.z))

        # align all the molecules
        for i, mol in enumerate(mols):
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))


        # draw grid image
        try:
            img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(200, 200))
            img.save(
                os.path.join(path, "{}_grid_image.png".format(path.split("/")[-1]))
            )
        except Chem.rdchem.KekulizeException:
            logger.info("Can't kekulize molecule")
        return mols


class NonMolecularVisualization:

    def __init__(self, dataset_name):
        self.is_tls = "tls" in dataset_name

    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()

        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

        rows, cols = np.where(adjacency_matrix >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adjacency_matrix[edge[0]][edge[1]]
            graph.add_edge(
                edge[0], edge[1], color=float(edge_type), weight=3 * edge_type
            )

        return graph

    def visualize_non_molecule(
        self,
        graph,
        pos,
        path,
        iterations=100,
        node_size=100,
        largest_component=False,
        time=None,
    ):
        if plt is None:
            logger.info("[Visualization] Matplotlib is unavailable; skip non-molecular plotting.")
            return
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        # Set node colors based on the eigenvectors
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        plt.figure()
        nx.draw(
            graph,
            pos,
            font_size=5,
            node_size=node_size,
            with_labels=False,
            node_color=U[:, 1],
            cmap=plt.cm.coolwarm,
            vmin=vmin,
            vmax=vmax,
            edge_color="grey",
        )
        if time is not None:
            plt.text(
                0.5,
                0.05,  # place below the graph
                f"t = {time:.2f}",
                ha="center",
                va="center",
                transform=plt.gcf().transFigure,
                fontsize=16,
            )

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(
        self, path: str, graphs: list, num_graphs_to_visualize: int, log="graph"
    ):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, "graph_{}.png".format(i))

            if self.is_tls:
                cg = CellGraph.from_dense_graph(graphs[i])
                cg.plot_graph(save_path=file_path, has_legend=True)
            else:
                graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
                self.visualize_non_molecule(graph=graph, pos=None, path=file_path)

            if plt is not None:
                im = plt.imread(file_path)
                try:
                    if swanlab.run and log is not None:
                        swanlab.log({log: swanlab.Image(im, caption=file_path)})
                except Exception as e:
                    logger.warning("[Visualization] swanlab image upload failed: %s", e)

    def visualize_chain(self, path, nodes_list, adjacency_matrix, times):

        graphs = []
        for i in range(nodes_list.shape[0]):
            if self.is_tls:
                graphs.append(
                    CellGraph.from_dense_graph((nodes_list[i], adjacency_matrix[i]))
                )
            else:
                graphs.append(self.to_networkx(nodes_list[i], adjacency_matrix[i]))

        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=0)

        

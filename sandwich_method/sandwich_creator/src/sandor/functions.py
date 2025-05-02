import os
import numpy as np
from ase.io import read, write, Trajectory
from ase.geometry import get_layers
from ase.visualize import view
from ase.build import make_supercell
from ase.atoms import Atoms
from scipy.signal import find_peaks
from ase.neighborlist import neighbor_list
from ase.cell import Cell
import pandas as pd
import matplotlib.pyplot as plt
from ase.geometry.analysis import Analysis

def get_stoichiometry(structure: Atoms) -> dict:
    
    chemical_symbols = structure.get_chemical_symbols()
    stoichiometry = {}

    for symbol in set(chemical_symbols):
        if symbol not in stoichiometry:
            
            stoichiometry[symbol] = len([x for x in chemical_symbols if x == symbol])

    return stoichiometry


def get_random_box(supercell: Atoms,
                   atom_ind: int,
                   windows: np.array,
                   origin_coords: np.array,
                   sandwich_cell: Cell) -> tuple:
    
    coords = supercell.get_positions()
    atom_coords = coords[atom_ind]

    new_coords = coords - atom_coords
    mask = np.abs(new_coords) < windows
    mask = np.all(mask, axis=1)

    substructure = Atoms(supercell[mask].get_chemical_symbols(),
                         new_coords[mask] + origin_coords,
                         pbc=True,
                         cell=sandwich_cell)
    
    substructure.set_velocities(supercell[mask].get_velocities())

    volume = np.prod(windows) * 8
    density = len(substructure) / volume
    coeff = density / len(supercell) * supercell.get_volume()
    
    return substructure, coeff


def get_metal_carbon_count(stoich: dict) -> tuple:
    """
    Getting counts of metal and carbon atoms.
    
    stoich - stoichiometry (dict).

    Return:
    tuple(int, int) - counts of metal and carbon atoms.
    
    """
    metal_count, carbon_count = (0, 0)
    
    for key in stoich.keys():
        if key != 'C':
            metal_count += stoich[key]
        
        else:
            carbon_count += stoich[key]
    
    return metal_count, carbon_count


def treat_solid_structure(path_to_xyz: str) -> tuple[Atoms, dict]:
    """"""
    
    structure = read(path_to_xyz)
    solid_stoichiometry = get_stoichiometry(structure)

    miller = (0, 0, 1)
    types, distances = get_layers(structure, miller, tolerance=5e-2)

    delta_cut = (distances[1:] - distances[0:-1]).max() / 2

    type_by_distance = {}

    for i in range(len(structure)):
        type_by_distance[i] = distances[types[i]]

    x_max, y_max, z_max = structure.cell.lengths()
    h_sol = (z_max / 4)
    min_limit = h_sol + delta_cut
    max_limit = z_max - h_sol - delta_cut
    all_distances = structure.get_all_distances()
    np.fill_diagonal(all_distances, all_distances.max())
    r_cut_solid = all_distances.min()

    h_liq_min = distances[np.argmin(np.abs(distances - h_sol))]
    h_liq_max = distances[np.argmin(np.abs(distances - (z_max - h_sol)))]

    first_layer_mask = types == np.where(distances == h_liq_min)[0][0]
    second_layer_mask = types == np.where(distances == h_liq_max)[0][0]

    h_liq_min = structure[first_layer_mask].get_positions()[::, 2].max()
    h_liq_max = structure[second_layer_mask].get_positions()[::, 2].min()

    mask = np.logical_or(distances < min_limit, distances > max_limit)

    mask_solid = np.array([False]*len(structure))

    for i in range(len(structure)):
        if type_by_distance[i] in distances[mask]:
            mask_solid[i] = True
    
    solid_sandwich = structure[mask_solid]

    return solid_sandwich, solid_stoichiometry, x_max, y_max, h_liq_min, h_liq_max, r_cut_solid



def treat_liquid_structure(path_to_xyz: str,
                            solid_sandwich: Atoms,
                            solid_stoichiometry: dict,
                            x_max: float,
                            y_max: float,
                            h_liq_min: float,
                            h_liq_max: float,
                            r_cut_solid: float) -> list[Atoms]:
    """"""

    liquid = read(path_to_xyz)
    liquid_stoichiometry = get_stoichiometry(liquid)
    assert solid_stoichiometry == liquid_stoichiometry

    P = np.diag([2, 2, 2])
    window_x, window_y, window_z = (x_max / 2,
                                    y_max / 2,
                                    (h_liq_max - h_liq_min) / 2)
    supercell = make_supercell(liquid, P)

    coords = supercell.get_positions()
    windows = np.array([window_x, window_y, window_z])
    cell = supercell.cell.lengths()

    mask = np.logical_and(coords > windows, coords < cell - windows)
    mask = np.all(mask, axis=1)

    size = 100
    random_inds = np.random.choice(np.where(mask)[0], size=size)

    params = []
    boxes = []

    z_origin = (h_liq_max - h_liq_min) / 2 + h_liq_min
    origin = np.array([x_max/2, y_max/2, z_origin])
    sandwich_cell = solid_sandwich.cell

    for ind in random_inds:
        box, param = get_random_box(supercell,
                                    ind,
                                    windows,
                                    origin,
                                    sandwich_cell)

        boxes.append(box)
        params.append(param)

    params = np.array(params)

    boxes_copy = boxes.copy()
    params_copy = params.copy()

    count = size // 100 * 10  # Приблизительно 10 %.
    order_mask = np.argsort(np.abs(params - 1))
    params = params_copy[order_mask[:count]]
    boxes = [boxes_copy[i] for i in order_mask[:count]]

    return boxes, windows



def get_sandwich_structures(
    path_to_solid_xyz: str,
    path_to_liquid_xyz: str
    ):
    """"""
    
    solid_sandwich, solid_stoichiometry, x_max, y_max, h_liq_min, h_liq_max, r_cut_solid = treat_solid_structure(path_to_solid_xyz)
    boxes, windows = treat_liquid_structure(path_to_liquid_xyz, solid_sandwich, solid_stoichiometry, x_max, y_max, h_liq_min, h_liq_max, r_cut_solid)

    
    substrucures = []
    masks = []

    for box in boxes:

        liquid_substructure = box

        liquid_velocities = liquid_substructure.get_velocities()
        elements = solid_sandwich.get_chemical_symbols()
        solid_velocities = solid_sandwich.get_velocities()
        solid_liquid_mask = [True] * len(solid_sandwich)
        max_solid_ind = len(solid_sandwich) - 1
        elements += liquid_substructure.get_chemical_symbols()
        solid_liquid_mask += [False] * len(liquid_substructure)
        solid_positions = solid_sandwich.get_positions()
        liquid_positions = liquid_substructure.get_positions()
        positions = np.concatenate([solid_positions, liquid_positions])
        cell = solid_sandwich.cell

        # first_layer = list(first_layer_mask) + [False] * len(liquid_substructure)
        # second_layer = list(second_layer_mask) + [False] * len(liquid_substructure)

        first_layer = np.logical_and(solid_positions[::, 2] <= h_liq_min, solid_positions[::, 2] >= h_liq_min - 3)
        first_layer = list(first_layer) + [False] * len(liquid_substructure)
        second_layer = np.logical_and(solid_positions[::, 2] >= h_liq_max, solid_positions[::, 2] <= h_liq_max + 3)
        second_layer = list(second_layer) + [False] * len(liquid_substructure)

        substrucure = Atoms(elements,
                            positions,
                            cell=cell,
                            pbc=True)

        # print(len(substrucure))


        substrucure.set_velocities(np.concatenate([solid_velocities, liquid_velocities]))

        solid_liquid_mask = np.array(solid_liquid_mask)

        # Check!!!

        # Плотность жидкой фазы в окне.
        p_win = (~solid_liquid_mask).sum() / (np.prod(windows) * 8)
        # print('p_win', round(p_win, 10))


        k = 0
        r_cut_coeff = 1

        # Пробегаем по всем атомам граничных поверхностей кристаллической фазы.
        # for ind_layer in np.where(np.logical_or(first_layer, second_layer))[0]:

        
        for ind_layer in np.where(solid_liquid_mask)[0]:

            liq_atom_positions = substrucure[~solid_liquid_mask].get_positions()
            layer_atom_position = substrucure[[ind_layer]].get_positions()[0]
            distances = (((liq_atom_positions - layer_atom_position)**2).sum(axis=1))**(0.5)
            mask_to_save = distances >= r_cut_solid * r_cut_coeff
            atom_to_delete_count = (~mask_to_save).sum()
            if atom_to_delete_count > 0:
                mask_to_save = np.concatenate([solid_liquid_mask[solid_liquid_mask], mask_to_save])
                substrucure = substrucure[mask_to_save]
                solid_liquid_mask = solid_liquid_mask[:-atom_to_delete_count]
                k += atom_to_delete_count

        # print('Количество удалённых атомов ->', k)

        # Плотность жидкой фазы после удаления атомов.
        p_after = (~solid_liquid_mask).sum() / ((-h_liq_min + h_liq_max) * x_max * y_max)
        # print('p_after', round(p_after, 4))

        # print('error', abs(p_win - p_after) / p_win)

        substrucures.append(substrucure)
        masks.append(solid_liquid_mask)
    
    
    
    # ===============================================================================================
    # Saving the proportion between metal and carbon atom counts.

    new_substrucures = []

    for ind in range(len(substrucures)):

        substrucure = substrucures[ind].copy()
        mask = masks[ind]

        stoich = get_stoichiometry(substrucure)

        metal_count, carbon_count = get_metal_carbon_count(stoich)

        if metal_count < carbon_count:
            # If metal atom count < carbon atom count, we will change carbons from liquid substructure
            # to metal atoms, and first we will replace the metal, which is less.
            
            for _ in range((carbon_count - metal_count) // 2):

                stoich = get_stoichiometry(substrucure)
                atom_type = [x for x in stoich.keys() if x != 'C']
                atom_count = [stoich[x] for x in atom_type]

                liquid_ind = np.where(~mask)[0]
                carbon_mask = [x == 'C' for x in substrucure[liquid_ind].get_chemical_symbols()]
                assert len(carbon_mask) > 0
                np.random.seed(12345)
                carbon_ind = np.random.choice(liquid_ind[carbon_mask])
                substrucure[carbon_ind].symbol = atom_type[np.argmin(atom_count)]

        elif metal_count > carbon_count:
            
            # If metal atom count Ю carbon atom count, we will change metal atoms from liquid substructure
            # to carbon ones, and first we will replace the metal, which is more.
            
            for _ in range(int(np.ceil((metal_count - carbon_count) / 2))):

                stoich = get_stoichiometry(substrucure)
                atom_type = [x for x in stoich.keys() if x != 'C']
                atom_count = [stoich[x] for x in atom_type]

                liquid_ind = np.where(~mask)[0]
                symbol = atom_type[np.argmax(atom_count)]
                metal_mask = [x == symbol for x in substrucure[liquid_ind].get_chemical_symbols()]
                assert len(metal_mask) > 0
                np.random.seed(12345)
                metal_ind = np.random.choice(liquid_ind[metal_mask])
                substrucure[metal_ind].symbol = 'C'
        
        stoich = get_stoichiometry(substrucure)
        metal_count, carbon_count = get_metal_carbon_count(stoich)

        # Checking the results.
        assert carbon_count - metal_count <= 1

        new_substrucures.append(substrucure)

    # ===============================================================================================
    
    
    
    return new_substrucures
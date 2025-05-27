# "formation_energy_per_atom", "band_gap" as default

import os
import random
import pickle
import argparse
from tqdm import tqdm
from dotenv import load_dotenv 

import numpy as np
from jarvis.db.figshare import data as load_jarvis_data
from mp_api.client import MPRester
from pymatgen.core import Structure

load_dotenv(verbose=True)

MAPI = os.getenv('MAPI')
TARGET_PROPERTIES = ["formation_energy_per_atom", "band_gap", "energy_above_hull"]
AVAILABLE_FIELDS = """
    ['builder_meta', 'nsites', 'elements', 'nelements', 'composition', 'composition_reduced', 'formula_pretty', 'formula_anonymous', 'chemsys', 'volume', 'density', 'density_atomic', 'symmetry', 'property_name', 'material_id', 'deprecated', 'deprecation_reasons', 'last_updated', 'origins', 'warnings', 'structure', 'task_ids', 'uncorrected_energy_per_atom', 'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull', 'is_stable', 'equilibrium_reaction_energy_per_atom', 'decomposes_to', 'xas', 'grain_boundaries', 'band_gap', 'cbm', 'vbm', 'efermi', 'is_gap_direct', 'is_metal', 'es_source_calc_id', 'bandstructure', 'dos', 'dos_energy_up', 'dos_energy_down', 'is_magnetic', 'ordering', 'total_magnetization', 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'num_magnetic_sites', 'num_unique_magnetic_sites', 'types_of_magnetic_species', 'bulk_modulus', 'shear_modulus', 'universal_anisotropy', 'homogeneous_poisson', 'e_total', 'e_ionic', 'e_electronic', 'n', 'e_ij_max', 'weighted_surface_energy_EV_PER_ANG2', 'weighted_surface_energy', 'weighted_work_function', 'surface_anisotropy', 'shape_factor', 'has_reconstructed', 'possible_species', 'has_props', 'theoretical', 'database_IDs']
"""


def fetch_structures_in_batches(api_key, total_limit=None, target_properties=[], chunk_size=500):
    from itertools import islice
    all_docs = []
    
    filter_kwargs = {
        "is_stable": True,
        "exclude_elements": ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]
    }

    with MPRester(api_key) as mpr:
        generator = mpr.materials.summary.search(
            **filter_kwargs,
            fields=["material_id", "structure", "formula_pretty"] + TARGET_PROPERTIES + target_properties,
            num_chunks=total_limit,
            chunk_size=chunk_size
        )

        for doc in islice(generator, total_limit):
            all_docs.append(doc)
    
    return all_docs

def mp_to_jarvis(mp_doc):
    struct: Structure = mp_doc.structure
    lattice = struct.lattice

    jarvis_dict = {
        'id': str(mp_doc.material_id),
        'formula': struct.composition.reduced_formula,
        'formation_energy_per_atom': mp_doc.formation_energy_per_atom,
        'band_gap': mp_doc.band_gap,
        'atoms': {
            'lattice_mat': [list(lattice.matrix[i]) for i in range(3)],
            'coords': [site.frac_coords.tolist() for site in struct],
            'elements': [str(site.specie) for site in struct],
            'abc': list(lattice.abc),
            'angles': list(lattice.angles),
            'cartesian': False,
            'props': ['' for _ in struct],
        }
    }

    return jarvis_dict

def merge_and_split_save(a, b, val_ratio=0.1, test_ratio=0.1, seed=42, root="",
                         train_path='train.pkl', val_path='val.pkl', test_path='test.pkl'):
    data = a + b
    
    print(f"Collected {len(data)} structures.")

    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    train_path = os.path.join(root, train_path)
    val_path = os.path.join(root, val_path)
    test_path = os.path.join(root, test_path)

    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    print("Saving train data...")

    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    print("Saving val data...")

    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)
    print("Saving test data...")

    print("Saving complete")
    return train_data, val_data, test_data

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--target',
        default=['formation_energy_per_atom']
    )

    parser.add_argument(
        '--max_sites',
        type=int,
        default=100
    )
    
    parser.add_argument(
        "--num_entries",
        type=int,
        default=None
    )
    
    args = parser.parse_args()
    
    SAVE_DIR = "data/mpjv"
    
    print(f"Fetching {args.num_entries if args.num_entries else 'all available'} structures")
    print(f"Target Properties: {args.target}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    jv_docs = load_jarvis_data("dft_3d", store_dir=SAVE_DIR)
    key_map = {
        "jid": "id",
        "formation_energy_peratom": "formation_energy_per_atom",
        "optb88vdw_bandgap": "band_gap"
    }

    jv_docs = [
        {key_map.get(k, k): v for k, v in d.items()}
        for d in jv_docs
    ]

    mp_docs = fetch_structures_in_batches(
        api_key=MAPI,
        total_limit=args.num_entries,
        target_properties=args.target,
        chunk_size=500
    )
    mp_docs = [mp_to_jarvis(doc) for doc in mp_docs]

    merge_and_split_save(mp_docs, jv_docs, root=SAVE_DIR)

if __name__ == "__main__":
    main()
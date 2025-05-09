# "formation_energy_per_atom", "band_gap" as default

import os
import warnings
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv 

import numpy as np
from jarvis.db.figshare import data as load_jarvis_data
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import (
    CrystalNN,
    MinimumDistanceNN,
    VoronoiNN
)

load_dotenv(verbose=True)

MAPI = os.getenv('MAPI')
SAVE_DIR = "data/processed"
RAW_DIR = "data/raw"
TARGET_PROPERTIES = ["formation_energy_per_atom", "band_gap"]
AVAILABLE_FIELDS = """
    ['builder_meta', 'nsites', 'elements', 'nelements', 'composition', 'composition_reduced', 'formula_pretty', 'formula_anonymous', 'chemsys', 'volume', 'density', 'density_atomic', 'symmetry', 'property_name', 'material_id', 'deprecated', 'deprecation_reasons', 'last_updated', 'origins', 'warnings', 'structure', 'task_ids', 'uncorrected_energy_per_atom', 'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull', 'is_stable', 'equilibrium_reaction_energy_per_atom', 'decomposes_to', 'xas', 'grain_boundaries', 'band_gap', 'cbm', 'vbm', 'efermi', 'is_gap_direct', 'is_metal', 'es_source_calc_id', 'bandstructure', 'dos', 'dos_energy_up', 'dos_energy_down', 'is_magnetic', 'ordering', 'total_magnetization', 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'num_magnetic_sites', 'num_unique_magnetic_sites', 'types_of_magnetic_species', 'bulk_modulus', 'shear_modulus', 'universal_anisotropy', 'homogeneous_poisson', 'e_total', 'e_ionic', 'e_electronic', 'n', 'e_ij_max', 'weighted_surface_energy_EV_PER_ANG2', 'weighted_surface_energy', 'weighted_work_function', 'surface_anisotropy', 'shape_factor', 'has_reconstructed', 'possible_species', 'has_props', 'theoretical', 'database_IDs']
"""

def get_nn_strategy(name="crystal"): 
    """
    | 전략 | 설명 | 장점 | 단점 |
    |------|------|------|------|
    | `CrystalNN` | 물리/화학적으로 가장 정교한 방법 (전하, 거리, 배위 고려) | 정확도 높음 | 느림, 복잡도 ↑ |
    | `VoronoiNN` | 공간적으로 Voronoi cell을 기준으로 이웃 정의 | 물리적 직관성 | 일부 구조에서 잘 작동 안 함 |
    """
    if name == "crystal":
        return CrystalNN()
    if name == "voronoi":
        return VoronoiNN()
    raise ValueError(f"Unknown NN strategy: {name}")

def fetch_structures_in_batches(api_key, total_limit=None, target_properties=[], chunk_size=500):
    from itertools import islice
    all_docs = []
    
    filter_kwargs = {
        "is_stable": True
    }

    with MPRester(api_key) as mpr:
        generator = mpr.materials.summary.search(
            **filter_kwargs,
            fields=["material_id", "structure"] + TARGET_PROPERTIES + target_properties,
            num_chunks=total_limit,
            chunk_size=chunk_size
        )

        for doc in islice(generator, total_limit):
            all_docs.append(doc)

    print(f"Collected {len(all_docs)} structures.")
    
    return all_docs

def get_existing_ids():
    if not os.path.exists(SAVE_DIR):
        return set()
    return {
        fname.replace(".json", "") for fname in os.listdir(SAVE_DIR)
        if fname.endswith(".json")
    }

def jarvis_atoms_to_structure(atoms_dict):
    """
    JARVIS 'atoms' dict --> pymatgen Structure
    """
    lattice = atoms_dict["lattice_mat"]
    coords = atoms_dict["coords"]
    species = atoms_dict["elements"]

    structure = Structure(
        lattice,
        species,
        coords,
        coords_are_cartesian=True
    )
    return structure

def process_jarvis_dataset(jarvis_list, existing_ids, mp_formulas=None, target_properties=None, nn_strategy=None, fallback_nn=None, max_num_sites=50):
    skipped = 0
    processed = 0
    pbar = tqdm(jarvis_list, desc="Processing JARVIS")

    for entry in pbar:
        jid = entry.get("jid")
        formula = entry.get("formula")
        
        if jid in existing_ids:
            skipped += 1
            continue
        if mp_formulas and formula in mp_formulas:
            skipped += 1
            continue

        try:
            structure = jarvis_atoms_to_structure(entry["atoms"])
            graph_data = structure_to_graph_data_with_fallback(
                structure,
                primary_nn=nn_strategy,
                fallback_nn=fallback_nn,
                max_num_sites=max_num_sites,
                pbc=True
            )

            props = {
                "formation_energy_per_atom": float(entry.get("formation_energy_peratom", "nan")),
                "band_gap": float(entry.get("optb88vdw_bandgap", "nan"))
            }

            save_as_json(graph_data, jid, props)
            processed += 1
        except Exception:
            skipped += 1
            continue

    print(f"[JARVIS] Processed {processed}, Skipped {skipped}")

def process_mp_dataset(api_key, existing_ids, nn_strategy, target_properties, num_entries=None, max_num_sites=50):
    """
    Materials Project 데이터를 불러오고 구조를 그래프로 변환해 저장
    """
    docs = fetch_structures_in_batches(
        api_key=api_key,
        total_limit=num_entries,
        target_properties=target_properties,
        chunk_size=500
    )

    mp_formulas = set()
    skipped = 0
    processed = 0

    pbar = tqdm(docs, desc="Processing MP", leave=False)
    for doc in pbar:
        material_id = doc.material_id
        if material_id in existing_ids:
            skipped += 1
            continue

        structure = doc.structure
        props = {
            key: getattr(doc, key, None)
            for key in TARGET_PROPERTIES
        }

        try:
            graph_data = structure_to_graph_data_with_fallback(
                structure,
                primary_nn=nn_strategy,
                fallback_nn=MinimumDistanceNN(),
                max_num_sites=max_num_sites,
                pbc=True
            )
            save_as_json(graph_data, material_id, props)
            mp_formulas.add(doc.formula_pretty)
            processed += 1
        except Exception:
            skipped += 1
            continue

    print(f"[MP] Processed {processed}, Skipped {skipped}")
    return mp_formulas

def structure_to_graph_data_with_fallback(structure, primary_nn, fallback_nn, max_num_sites=50, pbc=True):
    """
    Structure → 그래프 dict (노드/엣지 목록)
    """
    if len(structure.sites) > max_num_sites:
        #raise ValueError(f"Number of sites greater than {max_num_sites}")
        raise ValueError
    
    try:
        structure.add_oxidation_state_by_guess()
        s_graph = StructureGraph.from_local_env_strategy(structure, primary_nn)
    except UserWarning:
        structure.add_oxidation_state_by_guess()
        s_graph = StructureGraph.from_local_env_strategy(structure, fallback_nn)

    node_feats = [str(site.specie) for site in structure.sites]
    edges = []
    cart_vecs = []

    adjacency_dict = dict(s_graph.graph.adjacency())

    for i in adjacency_dict:
        for j in adjacency_dict[i]:
            if i < j:
                edges.append((i, j))
                
                frac_i = structure[i].frac_coords
                frac_j = structure[j].frac_coords
                d_frac = frac_j - frac_i

                # PBC
                if pbc:
                    d_frac -= np.round(d_frac)
                vec = structure.lattice.get_cartesian_coords(d_frac)
                
                cart_vecs.append(vec)

    if len(node_feats) < 2 or len(edges) == 0:
        #raise ValueError("less than 2 nodes or no edges found.")
        raise ValueError

    cart_vecs = np.array(cart_vecs)                    # shape: (N, 3)
    cart_dist = np.linalg.norm(cart_vecs, axis=1)    # shape: (N,)
    cart_dir = cart_vecs / cart_dist[:, None]       # shape: (N, 3)

    return {
        "nodes": node_feats,
        "edges": edges,
        "cart_dist": cart_dist.tolist(),
        "cart_dir": cart_dir.tolist()
    }


def save_as_json(graph_data, material_id, props):
    out_path = os.path.join(SAVE_DIR, f"{material_id}.json")

    payload = {
        "material_id": material_id,
        "graph": graph_data,
        "properties": props
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--nn_strategy",
        type=str,
        default="crystal",
        choices=["crystal", "voronoi"],
    )
    parser.add_argument(
        "--num_entries",
        type=int,
        default=None
    )
    parser.add_argument(
        '--target', 
        nargs='+',
        default=['formation_energy_per_atom', 'band_gap'],
        help='Target properties (default: formation_energy_per_atom band_gap)'
    )

    parser.add_argument(
        '--max_sites',
        type=int,
        default=50
    )
    
    args = parser.parse_args()
    # CGCNN radius not defined
    warnings.simplefilter("error", UserWarning)

    print(f"Fetching {args.num_entries if args.num_entries else 'all available'} structures using '{args.nn_strategy}' strategy")
    print(f"Target Properties: {args.target}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    existing_ids = get_existing_ids()

    nn_strategy = get_nn_strategy(args.nn_strategy)
    
    mp_formulas = process_mp_dataset(
        api_key=MAPI,
        existing_ids=existing_ids,
        nn_strategy=nn_strategy,
        target_properties=args.target,
        num_entries=args.num_entriesx,
        max_num_sites=args.max_sites
    )
    
    jarvis_data = load_jarvis_data("dft_3d", store_dir=RAW_DIR)
    print(f"Loaded {len(jarvis_data)} JARVIS entries")

    process_jarvis_dataset(
        jarvis_list=jarvis_data,
        existing_ids=existing_ids,
        mp_formulas=mp_formulas,
        target_properties=args.target,
        nn_strategy=nn_strategy,
        fallback_nn=MinimumDistanceNN(),
        max_num_sites=args.max_sites
    )


if __name__ == "__main__":
    main()
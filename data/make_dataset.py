import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv 

from mp_api.client import MPRester
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import (
    CrystalNN,
    MinimumDistanceNN,
    VoronoiNN
)

load_dotenv(verbose=True)

MAPI = os.getenv('MAPI')
SAVE_DIR = "data/processed"
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

def fetch_structures_in_batches(api_key, total_limit=None, chunk_size=500):
    from itertools import islice
    all_docs = []
    
    filter_kwargs = {
        "is_stable": True
    }

    with MPRester(api_key) as mpr:
        generator = mpr.materials.summary.search(
            **filter_kwargs,
            fields=["material_id", "structure"] + TARGET_PROPERTIES,
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


def structure_to_graph_data_with_fallback(structure, primary_nn, fallback_nn):
    """
    Structure → 그래프 dict (노드/엣지 목록)
    """
    try:
        structure.add_oxidation_state_by_guess()
        s_graph = StructureGraph.from_local_env_strategy(structure, primary_nn)
    except Exception:
        structure.add_oxidation_state_by_guess()
        s_graph = StructureGraph.from_local_env_strategy(structure, fallback_nn)

    node_feats = [str(site.specie) for site in structure.sites]
    edges = []

    adjacency_dict = dict(s_graph.graph.adjacency())

    for i in adjacency_dict:
        for j in adjacency_dict[i]:
            if i < j:
                edges.append((i, j))

    return {
        "nodes": node_feats,
        "edges": edges
    }


def save_as_json(graph_data, material_id, props):
    os.makedirs(SAVE_DIR, exist_ok=True)
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
    args = parser.parse_args()

    print(f"Fetching {args.num_entries if args.num_entries else 'all available'} structures using '{args.nn_strategy}' strategy")
    
    existing_ids = get_existing_ids()

    docs = fetch_structures_in_batches(
        api_key=MAPI,
        total_limit=args.num_entries,
        chunk_size=500
    )

    nn_strategy = get_nn_strategy(args.nn_strategy)
    
    skipped_materials_cnt = 0
    pbar = tqdm(docs, leave=False)
    for doc in pbar:
        material_id = doc.material_id
        if material_id in existing_ids:
            # print(f"{material_id} already exists.")
            skipped_materials_cnt += 1
            continue
        pbar.set_description(f"Processing {material_id}")
        structure = doc.structure

        props = {
            key: getattr(doc, key, None)
            for key in TARGET_PROPERTIES
        }

        try:
            graph_data = structure_to_graph_data_with_fallback(
                structure,
                primary_nn=nn_strategy,
                fallback_nn=MinimumDistanceNN()
            )
            save_as_json(graph_data, material_id, props)
        except Exception as e:
            print(f"Failed for {material_id}: {e}")

    print(f"Processed {len(docs) - skipped_materials_cnt}, Skipped {skipped_materials_cnt} materials.")

if __name__ == "__main__":
    main()
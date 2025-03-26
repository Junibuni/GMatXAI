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

def get_nn_strategy(name="crystal"): 
    """
    | 전략 | 설명 | 장점 | 단점 |
    |------|------|------|------|
    | `CrystalNN` | 물리/화학적으로 가장 정교한 방법 (전하, 거리, 배위 고려) | 정확도 높음 | 느림, 복잡도 ↑ |
    | `MinimumDistanceNN` | 기준 거리 내 가장 가까운 원자들만 연결 | 빠름, 매우 간단 | 정확도 낮음, 과소연결 가능 |
    | `VoronoiNN` | 공간적으로 Voronoi cell을 기준으로 이웃 정의 | 물리적 직관성 | 일부 구조에서 잘 작동 안 함 |
    """
    if name == "crystal":
        return CrystalNN()
    if name == "min":
        return MinimumDistanceNN()
    if name == "voronoi":
        return VoronoiNN()
    raise ValueError(f"Unknown NN strategy: {name}")

def fetch_structures(api_key, num_entries=100):
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            is_stable=True,
            fields=["material_id", "structure"] + TARGET_PROPERTIES,
            num_chunks=1,
            chunk_size=num_entries
        )
    return docs


def structure_to_graph_data(structure, nn_strategy):
    """
    Structure → 그래프 dict (노드/엣지 목록)
    """
    structure.add_oxidation_state_by_guess()
    try:
        s_graph = StructureGraph.from_local_env_strategy(structure, nn_strategy)
    except Exception as e:
        raise RuntimeError(f"StructureGraph conversion failed: {e}")

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
        choices=["crystal", "min", "voronoi"],
    )
    parser.add_argument(
        "--num_entries",
        type=int,
        default=200
    )
    args = parser.parse_args()

    print(f"Fetching {args.num_entries} structures using '{args.nn_strategy}' strategy")
    
    docs = fetch_structures(MAPI, num_entries=200)
    nn_strategy = get_nn_strategy(args.nn_strategy)
    
    for doc in tqdm(docs):
        material_id = doc.material_id
        structure = doc.structure
        structure.add_oxidation_state_by_guess()

        props = {
            key: getattr(doc, key, None)
            for key in TARGET_PROPERTIES
        }

        try:
            graph_data = structure_to_graph_data(structure, nn_strategy)
            save_as_json(graph_data, material_id, props)
        except Exception as e:
            print(f"Failed for {material_id}: {e}")


if __name__ == "__main__":
    main()
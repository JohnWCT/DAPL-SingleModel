"""
B_precontext.py
===============

用途
----
新版 drug precontext 流程。此檔案保留原始 `precontext.py` 的
SMILES -> molecular graph -> GIN substructure/context pretraining 概念，
但改為讀取 Winnie 新版 drug input：

- `data_Winnie/drug_smiles.csv`

資料格式
--------
預設欄位：

- `broad_id`：drug ID，會作為輸出 latent dictionary 的 key。
- `smiles`：SMILES 字串，會轉成 molecular graph。
- `name`：drug name，用於後續 `C_prototypical.py` 將 TCGA `drug_name`
  對齊回 `broad_id`。

程式會檢查：

- drug ID 是否重複。
- SMILES 是否可被 RDKit 解析。
- molecular graph 是否有效。

Drug latent representation 定義
-------------------------------
訓練完成後，使用完整 drug graph 通過 `GINConvNet` 的輸出作為 drug latent
representation。輸出 pkl 格式為：

    {
        "BRD-...": [latent_value_1, latent_value_2, ...],
        "BRD-...": [latent_value_1, latent_value_2, ...],
    }

輸出檔案
--------
`--output_dir` 內會產生：

- `drug_latent_representation.pkl`
- `drug_encoder.pth`
- `B_training_log.txt`
- `invalid_smiles.csv`，只有遇到無效 SMILES 時才會產生。

Docker 執行範例
---------------
請在 Docker container `DAPL` 內執行，不要在本機 Python 環境安裝或修改套件：

    docker exec DAPL python3 /workspace/DAPL-master/B_precontext.py \
      --drug_input /workspace/DAPL-master/data_Winnie/drug_smiles.csv \
      --drug_id_col broad_id \
      --smiles_col smiles \
      --drug_name_col name \
      --output_dir /workspace/DAPL-master/output_dir/B_precontext \
      --random_seed 42

快速 smoke test 可降低 epoch：

    docker exec DAPL python3 /workspace/DAPL-master/B_precontext.py \
      --epochs 1 \
      --patience 1 \
      --output_dir /workspace/DAPL-master/output_dir/B_precontext_smoke

後續銜接
--------
`C_prototypical.py` 會使用本檔輸出的：

    /workspace/DAPL-master/output_dir/B_precontext/drug_latent_representation.pkl
"""

import argparse
import os
import pickle
import random
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch_geometric.data as DATA
from rdkit import Chem
from sklearn.metrics import roc_auc_score

from drugmodels.ginconv import GINConvNet
from tools.dataprocess import atom_features


device = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_message(log_path, message):
    print(message)
    with open(log_path, "a") as handle:
        handle.write(str(message) + "\n")


def cycle_index(num, shift):
    arr = torch.arange(num, device=device) + shift
    arr[-shift:] = torch.arange(shift, device=device)
    return arr


def graph_data_obj_to_nx_simple(data):
    graph = nx.Graph()
    atom_features_np = data.x.cpu().numpy()
    for idx in range(atom_features_np.shape[0]):
        graph.add_node(idx, atom_num_idx=atom_features_np[idx])
    edge_index = data.edge_index.cpu().numpy()
    for edge_pos in range(0, edge_index.shape[1], 2):
        begin_idx = int(edge_index[0, edge_pos])
        end_idx = int(edge_index[1, edge_pos])
        if not graph.has_edge(begin_idx, end_idx):
            graph.add_edge(begin_idx, end_idx)
    return graph


def nx_to_graph_data_obj_simple(graph):
    atom_features_list = [node["atom_num_idx"] for _, node in graph.nodes(data=True)]
    x = torch.tensor(np.asarray(atom_features_list), dtype=torch.float)
    edges = []
    for begin_idx, end_idx in graph.edges():
        edges.append((begin_idx, end_idx))
        edges.append((end_idx, begin_idx))
    if edges:
        edge_index = torch.tensor(np.asarray(edges).T, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return DATA.Data(x=x, edge_index=edge_index)


def reset_idxes(graph):
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping, copy=True), mapping


class ExtractSubstructureContextPair:
    def __init__(self, k, l1, l2):
        self.k = -1 if k == 0 else k
        self.l1 = -1 if l1 == 0 else l1
        self.l2 = -1 if l2 == 0 else l2

    def __call__(self, data, root_idx=None):
        data.x_context = None
        num_atoms = data.x.size()[0]
        if num_atoms == 0:
            return data
        if root_idx is None:
            root_idx = random.sample(range(num_atoms), 1)[0]
        graph = graph_data_obj_to_nx_simple(data)
        substruct_node_idxes = nx.single_source_shortest_path_length(graph, root_idx, self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_graph = graph.subgraph(substruct_node_idxes)
            substruct_graph, substruct_node_map = reset_idxes(substruct_graph)
            substruct_data = nx_to_graph_data_obj_simple(substruct_graph)
            data.x_substruct = substruct_data.x
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[root_idx]])
        l1_node_idxes = nx.single_source_shortest_path_length(graph, root_idx, self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(graph, root_idx, self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_graph = graph.subgraph(context_node_idxes)
            context_graph, context_node_map = reset_idxes(context_graph)
            context_data = nx_to_graph_data_obj_simple(context_graph)
            data.x_context = context_data.x
            data.edge_index_context = context_data.edge_index
            overlap_idxes = list(set(context_node_idxes).intersection(set(substruct_node_idxes)))
            if overlap_idxes:
                overlap_reorder = [context_node_map[old_idx] for old_idx in overlap_idxes]
                data.overlap_context_substruct_idx = torch.tensor(overlap_reorder)
        return data


class BatchSubstructContext(DATA.Data):
    @staticmethod
    def from_data_list(data_list):
        batch = BatchSubstructContext()
        keys = [
            "center_substruct_idx",
            "edge_index_substruct",
            "x_substruct",
            "overlap_context_substruct_idx",
            "edge_index_context",
            "x_context",
        ]
        for key in keys:
            batch[key] = []
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []
        cumsum_substruct = 0
        cumsum_context = 0
        kept_idx = 0
        for data in data_list:
            if not hasattr(data, "x_context") or data.x_context is None:
                continue
            if not hasattr(data, "overlap_context_substruct_idx"):
                continue
            num_nodes_substruct = len(data.x_substruct)
            num_nodes_context = len(data.x_context)
            batch.batch_overlapped_context.append(
                torch.full((len(data.overlap_context_substruct_idx),), kept_idx, dtype=torch.long)
            )
            batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))
            for key in ["center_substruct_idx", "edge_index_substruct", "x_substruct"]:
                item = data[key]
                if key in ["edge_index_substruct", "center_substruct_idx"]:
                    item = item + cumsum_substruct
                batch[key].append(item)
            for key in ["overlap_context_substruct_idx", "edge_index_context", "x_context"]:
                item = data[key]
                if key in ["overlap_context_substruct_idx", "edge_index_context"]:
                    item = item + cumsum_context
                batch[key].append(item)
            cumsum_substruct += num_nodes_substruct
            cumsum_context += num_nodes_context
            kept_idx += 1
        for key in keys:
            batch[key] = torch.cat(batch[key], dim=-1 if "edge_index" in key else 0)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)
        return batch.contiguous()


class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=16, shuffle=True, **kwargs):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list),
            **kwargs,
        )


def smiles_to_pyg(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"invalid SMILES: {smiles}")
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        edges.append([begin_idx, end_idx])
        edges.append([end_idx, begin_idx])
    x = torch.tensor(np.asarray(features), dtype=torch.float, device=device)
    if edges:
        edge_index = torch.tensor(np.asarray(edges).T, dtype=torch.long, device=device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    return DATA.Data(x=x, edge_index=edge_index)


def read_drugs(path, drug_id_col, smiles_col, drug_name_col):
    df = pd.read_csv(path)
    required = [drug_id_col, smiles_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing required drug columns in {path}: {missing}")
    cols = [drug_id_col, smiles_col] + ([drug_name_col] if drug_name_col and drug_name_col in df.columns else [])
    df = df[cols].dropna(subset=[drug_id_col, smiles_col]).copy()
    df[drug_id_col] = df[drug_id_col].astype(str)
    if df[drug_id_col].duplicated().any():
        dup_count = int(df[drug_id_col].duplicated().sum())
        unique_dup_ids = int(df.loc[df[drug_id_col].duplicated(), drug_id_col].nunique())
        print(
            f"[read_drugs] found duplicated drug IDs ({dup_count} rows, {unique_dup_ids} unique IDs); "
            "keep first row per drug_id"
        )
        # Keep one canonical row per drug_id so latent dictionary key remains unique.
        df = df.drop_duplicates(subset=[drug_id_col], keep="first").reset_index(drop=True)
    return df


def train_epoch(model, context_model, loader, optimizer_substruct, optimizer_context, criterion):
    model.train()
    context_model.train()
    loss_accum = 0.0
    auc_accum = 0.0
    steps = 0
    for batch in loader:
        batch = batch.to(device)
        if not hasattr(batch, "overlapped_context_size") or len(batch.overlapped_context_size) < 2:
            continue
        subdata = DATA.Data().to(device)
        subdata.x, subdata.edge_index = batch.x_substruct.float(), batch.edge_index_substruct
        substruct_rep = model(subdata)
        substruct_rep = substruct_rep[batch.center_substruct_idx]
        contextdata = DATA.Data().to(device)
        contextdata.x, contextdata.edge_index = batch.x_context.float(), batch.edge_index_context
        overlapped_node_rep = context_model(contextdata)
        overlapped_node_rep = overlapped_node_rep[batch.overlap_context_substruct_idx]
        expanded_substruct_rep = torch.cat(
            [substruct_rep[i].repeat((batch.overlapped_context_size[i], 1)) for i in range(len(substruct_rep))],
            dim=0,
        )
        pred_pos = torch.sum(expanded_substruct_rep * overlapped_node_rep, dim=1)
        if len(substruct_rep) < 2:
            continue
        shifted_substruct_rep = substruct_rep[cycle_index(len(substruct_rep), 1)]
        shifted_expanded = torch.cat(
            [shifted_substruct_rep[i].repeat((batch.overlapped_context_size[i], 1)) for i in range(len(shifted_substruct_rep))],
            dim=0,
        )
        pred_neg = torch.sum(shifted_expanded * overlapped_node_rep, dim=1)
        loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos), device=device).double())
        loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg), device=device).double())
        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()
        loss = loss_pos + loss_neg
        loss.backward()
        optimizer_substruct.step()
        optimizer_context.step()
        loss_accum += float(loss.detach().cpu().item())
        try:
            auc_accum += roc_auc_score(
                torch.cat((torch.ones(len(pred_pos)), torch.zeros(len(pred_neg))), dim=0).numpy(),
                torch.cat((pred_pos.detach().cpu(), pred_neg.detach().cpu()), dim=0).numpy(),
            )
        except ValueError:
            auc_accum += np.nan
        steps += 1
    if steps == 0:
        raise RuntimeError(
            "No valid training steps in this epoch. "
            "Check batch_size and valid substructure/context pair count."
        )
    return loss_accum / steps, auc_accum / steps


def pretrain_gin(graphs, args, log_path):
    if len(graphs) < 2:
        raise ValueError("valid drug graph count must be >= 2 for context pretraining")
    model = GINConvNet(input_dim=graphs[0].x.shape[1], output_dim=args.latent_dim, pretrain_flag=True).to(device)
    context_model = GINConvNet(input_dim=graphs[0].x.shape[1], output_dim=args.latent_dim, pretrain_flag=True).to(device)
    optimizer_substruct = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_context = optim.Adam(context_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    transform = ExtractSubstructureContextPair(args.k, args.l1, args.l2)
    transformed = [transform(graph.clone().cpu()) for graph in graphs]
    filtered = [
        graph
        for graph in transformed
        if hasattr(graph, "x_context")
        and graph.x_context is not None
        and hasattr(graph, "overlap_context_substruct_idx")
        and len(graph.overlap_context_substruct_idx) > 0
    ]
    if not filtered:
        raise ValueError("no valid substructure/context graph pairs were created")
    loader = DataLoaderSubstructContext(dataset=filtered, batch_size=args.batch_size, shuffle=True)
    best_state = deepcopy(model.state_dict())
    min_loss = float("inf")
    tolerance = 0
    for epoch in range(args.epochs):
        train_loss, train_auc = train_epoch(model, context_model, loader, optimizer_substruct, optimizer_context, criterion)
        log_message(log_path, {"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc})
        if train_loss < min_loss:
            min_loss = train_loss
            best_state = deepcopy(model.state_dict())
            tolerance = 0
        else:
            tolerance += 1
        if tolerance >= args.patience:
            log_message(log_path, f"GIN early stopping at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    return model


def encode_drugs(model, drug_ids, graphs):
    model.eval()
    latents = {}
    with torch.no_grad():
        for drug_id, graph in zip(drug_ids, graphs):
            graph = graph.to(device)
            latent = model(graph).detach().cpu().numpy()
            if latent.ndim == 2:
                latent = latent.mean(axis=0)
            else:
                latent = latent.reshape(-1)
            latents[str(drug_id)] = latent.tolist()
    return latents


def validate_latent_dims(latent_dict):
    dims = sorted({len(v) for v in latent_dict.values()})
    if len(dims) != 1:
        raise ValueError(f"inconsistent latent dimensions detected: {dims}")
    return dims[0]


def export_drug_latents(model, valid_ids, graphs, output_dir):
    latent_dict = encode_drugs(model, valid_ids, graphs)
    latent_dim = validate_latent_dims(latent_dict)
    with open(os.path.join(output_dir, "drug_latent_representation.pkl"), "wb") as handle:
        pickle.dump(latent_dict, handle)
    return latent_dim


def parse_args():
    parser = argparse.ArgumentParser("DAPL B_precontext drug precontext")
    parser.add_argument("--drug_input", default="data_Winnie/drug_smiles.csv")
    parser.add_argument("--drug_id_col", default="broad_id")
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--drug_name_col", default="name")
    parser.add_argument("--output_dir", default="output_dir/B_precontext")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--l1", type=int, default=1)
    parser.add_argument("--l2", type=int, default=7)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--encoder_pth", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.random_seed)
    ensure_dir(args.output_dir)
    log_path = os.path.join(args.output_dir, "B_training_log.txt")
    log_message(log_path, f"device={device}")
    if args.batch_size < 2:
        raise ValueError("--batch_size must be >= 2 for valid negative sampling")
    drug_df = read_drugs(args.drug_input, args.drug_id_col, args.smiles_col, args.drug_name_col)
    graphs = []
    valid_ids = []
    invalid_rows = []
    for _, row in drug_df.iterrows():
        try:
            graphs.append(smiles_to_pyg(row[args.smiles_col]))
            valid_ids.append(str(row[args.drug_id_col]))
        except Exception as err:
            invalid_rows.append({"drug_id": row[args.drug_id_col], "error": str(err)})
    if invalid_rows:
        pd.DataFrame(invalid_rows).to_csv(os.path.join(args.output_dir, "invalid_smiles.csv"), index=False)
        log_message(log_path, f"excluded invalid SMILES rows: {len(invalid_rows)}")
    if not graphs:
        raise ValueError("no valid drug graphs available for training")
    total_drugs = len(drug_df)
    invalid_count = len(invalid_rows)
    valid_count = len(graphs)
    excluded_ratio = invalid_count / max(1, total_drugs)
    log_message(
        log_path,
        f"drug_summary total={total_drugs}, valid={valid_count}, invalid={invalid_count}, excluded_ratio={excluded_ratio:.4f}",
    )
    if valid_count < 2:
        raise ValueError("valid drug graph count must be >= 2")
    log_message(log_path, f"valid_drugs={len(graphs)}, node_feature_dim={graphs[0].x.shape[1]}")
    model = GINConvNet(input_dim=graphs[0].x.shape[1], output_dim=args.latent_dim, pretrain_flag=True).to(device)
    if args.skip_train:
        if not args.encoder_pth:
            raise ValueError("--skip_train requires --encoder_pth")
        model.load_state_dict(torch.load(args.encoder_pth, map_location=device))
        log_message(log_path, f"loaded existing encoder from {args.encoder_pth}")
    else:
        model = pretrain_gin(graphs, args, log_path)
        torch.save(model.state_dict(), os.path.join(args.output_dir, "drug_encoder.pth"))
    latent_dim = export_drug_latents(model, valid_ids, graphs, args.output_dir)
    log_message(log_path, f"exported drug latents with fixed_dim={latent_dim}")
    log_message(log_path, "B_precontext completed")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json
import numpy as np
from scipy import sparse
from tqdm import tqdm

AMINO20 = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AMINO20)
AA_IDX = {a:i for i,a in enumerate(AMINO20)}

def parse_fasta(fp):
    header = None
    seq_chunks = []
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks).upper()
                header = line
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        yield header, "".join(seq_chunks).upper()

def clean_sequence(seq: str) -> str:
    return "".join([c for c in seq if c in AA_SET])

def label_from_header(header: str, label_regex: str = None) -> str:
    import re
    if label_regex:
        m = re.search(label_regex, header)
        if m: return m.group(0)
    head = header[1:].strip()
    return head.split()[0]

def pair_index(a: str, b: str) -> int:
    return AA_IDX[a]*20 + AA_IDX[b]

def quad_index(p1: int, p2: int) -> int:
    return p1*400 + p2

def build_feature_indices(seq: str, skip_x: int):
    seq = clean_sequence(seq)
    n = len(seq)
    p2_off = 2 + skip_x
    win_len = p2_off + 2
    present = set()
    for i in range(0, n - win_len + 1):
        a1, a2 = seq[i], seq[i+1]
        b1, b2 = seq[i+p2_off], seq[i+p2_off+1]
        if a1 in AA_SET and a2 in AA_SET and b1 in AA_SET and b2 in AA_SET:
            p1 = pair_index(a1, a2)
            p2 = pair_index(b1, b2)
            present.add(quad_index(p1, p2))
    return sorted(present)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out-features", required=True)
    ap.add_argument("--out-labels", required=True)
    ap.add_argument("--x-skip", type=int, default=0)
    ap.add_argument("--max-seqs", type=int, default=300)
    ap.add_argument("--label-regex", default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_features), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_labels), exist_ok=True)

    rows, cols, data = [], [], []
    labels, headers = [], []
    rcount = 0
    for header, seq in tqdm(parse_fasta(args.fasta), desc="FASTA", unit="seq"):
        feat_idx = build_feature_indices(seq, args.x_skip)
        for c in feat_idx:
            rows.append(rcount); cols.append(int(c)); data.append(1)
        labels.append(label_from_header(header, args.label_regex))
        headers.append(header)
        rcount += 1
        if rcount >= args.max_seqs: break

    from scipy import sparse
    X = sparse.csr_matrix((data, (rows, cols)), shape=(rcount, 160000), dtype=np.uint8)
    sparse.save_npz(args.out_features, X)
    with open(args.out_labels, "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "headers": headers}, f, ensure_ascii=False, indent=2)

    print(f"Saved features to {args.out_features}, shape={X.shape}, nnz={X.nnz}")
    print(f"Saved labels to {args.out_labels}, n={len(labels)}")

if __name__ == "__main__":
    main()

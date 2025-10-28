
# run_fast_windows.ps1
$ErrorActionPreference = "Stop"
$FASTA = "data\astral_sample.txt"
$FEAT  = "data\features_sample.npz"
$LAB   = "data\labels_sample.json"
$OUT   = "results\sample_fast"

python scripts\extract_features_fast.py --fasta $FASTA --out-features $FEAT --out-labels $LAB --x-skip 0 --max-seqs 300
python scripts\pca_and_cluster_fast.py --features $FEAT --labels $LAB --out-dir $OUT --k-range 2:10 --n-components 150

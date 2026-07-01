#!/bin/bash
set -e

if [[ -z "$1" || -z "$2" ]]; then
  echo "Usage: $0 input_dir output_dir"
  exit 1
fi

for batch_dir in "$1"/batch_*; do
  batch_name=$(basename "$batch_dir")
  out_dir="$2"/"${batch_name}"
  mkdir -p "$out_dir"

  echo "=== Procesando $batch_dir → $out_dir ==="

  # Si ya hay resultados en esa carpeta, saltar (por si ya lo hiciste)
  if ls "$out_dir"/*.json >/dev/null 2>&1; then
    echo "  Resultados ya existen, salto este batch."
    continue
  fi

  /home/jovyan/conda_envs/alphafold3/bin/python /home/jovyan/TFG/run_mmseqs_retry.py --input_dir "$batch_dir" --output_dir "$out_dir" --templates --num_templates 4 #2>&1 | tee run_Efectores_mmseqs.log
  
done

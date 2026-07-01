#!/bin/bash
set -e

if [[ -z "$1" || -z "$2" ]]; then
  echo "Usage: $0 input_dir output_dir"
  exit 1
fi

# Crear una carpeta temporal limpia para los inputs individuales
TMP_IN_DIR="/home/jovyan/TFG/tmp_af3_input"
mkdir -p "$TMP_IN_DIR"

for batch_dir in $1/batch_*; do
  batch_name=$(basename "$batch_dir")
  out_dir="$2/${batch_name}"
  mkdir -p "$out_dir"

  echo "=== Procesando $batch_dir → $out_dir ==="

  # Si ya hay resultados en esa carpeta, saltar (por si ya lo hiciste)
  for pair_sequences_mmseqs in "$1"/"$batch_name"/*; do
    echo "Secuencia de partida: $pair_sequences_mmesqs"
    pair_sequences=${pair_sequences_mmseqs%_mmseqs.json}
    pair_sequences=$(basename "$pair_sequences")
    
    echo "=== Procesando $pair_sequences ==="

    echo "Buscando si está bien completado $out_dir/$pair_sequences/"
    
    if [ -f "$out_dir"/"$pair_sequences"/"TERMS_OF_USE.md" ]; then
      echo "  Resultados ya existen, salto la pareja $pair_sequences."
      continue
    fi

    # LIMPIEZA Y PREPARACIÓN DE LA CARPETA TEMPORAL
    rm -rf "$TMP_IN_DIR"/*
    # Copiamos el archivo con extensión .json limpia para asegurar que AF3 lo vea
    cp "$pair_sequences_mmseqs" "$TMP_IN_DIR"

    # Work around XLA issue causing compilation time to greatly increase
    export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
    # Do not use unified memory, to be faster:
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_CLIENT_MEM_FRACTION=0.95


    AF3_ENV_PATH="/home/jovyan/conda_envs/alphafold3/bin"

    $AF3_ENV_PATH/python /home/jovyan/TFG/alphafold3/run_alphafold.py --input_dir="$TMP_IN_DIR" --model_dir=/home/jovyan/TFG/alphafold3_parameters --output_dir=$out_dir --norun_data_pipeline --save_embeddings

  done
  
done

# Limpieza final
rm -rf "$TMP_IN_DIR"

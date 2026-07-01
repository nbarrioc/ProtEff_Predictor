#!/bin/bash

# Script created by Nerea Barrio Cabezas
# March, 2026

if [[ -z "$1" || -z "$2" ]]; then
  echo "Usage: $0 input_dir output_file temporary_dir"
  exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2

mkdir -p "$3"

# Group together in one fasta file all the individual fasta
for f in $INPUT_DIR/*.fasta; do
    # Take only the lines that are not empty
    grep -v '^[[:space:]]*$' "$f" >> all_fasta_seqs.fasta
    echo "" >> all_fasta_seqs.fasta
done

# Execute MMSEQS
mmseqs easy-cluster all_fasta_seqs.fasta "$OUTPUT_DIR" tmp \
    --min-seq-id 0.3 \
    -c 0.8 \
    --cov-mode 0 \
    --threads 8

# Remove the sequences file
rm all_fasta_seqs.fasta
rm -rf "$3"

echo "Proceso terminado. Todas las secuencias han sido agrupadas en el directorio mmseqs_results_similarity_30."
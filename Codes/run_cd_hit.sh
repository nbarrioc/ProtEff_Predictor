#!/bin/bash

# Script created by Nerea Barrio Cabezas
# March, 2026

if [[ -z "$1" || -z "$2" ]]; then
  echo "Usage: $0 input_dir output_file"
  exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2

# Group together in one fasta file all the individual fasta
for f in $INPUT_DIR/*.fasta; do
    # Take only the lines that are not empty
    grep -v '^[[:space:]]*$' "$f" >> all_fasta_seqs.fasta
    echo "" >> all_fasta_seqs.fasta
done

# Execute CD-HIT
cd-hit -i all_fasta_seqs.fasta -o "$2" -c 0.4 -n 2 -G 0 -aS 0.8 -M 8000 -T 8

# Remove the sequences file
rm all_fasta_seqs.fasta

echo "Proceso terminado. Todas las secuencias han sido agrupadas en el directorio cd_hit."
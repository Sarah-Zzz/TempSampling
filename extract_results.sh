#!/bin/bash

# Check if input file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

INPUT_FILE=$1

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File $INPUT_FILE not found."
    exit 1
fi

# Extract fields
# 'epoch': extract number after it
EPOCH=$(grep -o "'epoch': [0-9]*" "$INPUT_FILE" | head -n 1 | awk '{print $2}')

# Total training time: extract number after it
TRAIN_TIME=$(grep -o "Total training time:[0-9.]*" "$INPUT_FILE" | tail -n 1 | cut -d':' -f2)

# total_edges_sampled: extract number after it
EDGES=$(grep -o "total_edges_sampled: [0-9]*" "$INPUT_FILE" | tail -n 1 | awk '{print $2}')

# Best AUC: extract number after it
AUC=$(grep -o "Best AUC:[0-9.]*" "$INPUT_FILE" | tail -n 1 | cut -d':' -f2)

# ave val loss: extract number after it
# "ave val loss" are three words, so the number is the 4th field in the matched string
LOSS=$(grep -o "ave val loss [0-9.]*" "$INPUT_FILE" | tail -n 1 | awk '{print $4}')

# Print results
# echo "Epoch: $EPOCH"
# echo "Total Training Time: $TRAIN_TIME"
# echo "Total Edges Sampled: $EDGES"
# echo "Best AUC: $AUC"
# echo "Ave Val Loss: $LOSS"
echo "$EPOCH"
echo
echo "$TRAIN_TIME"
echo "$EDGES"
echo "$AUC"
echo "$LOSS"
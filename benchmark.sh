#!/bin/bash

# circuit_type_list=(supremacy hwea bv qft aqft adder regular erdos)
circuit_type_list=(regular)
for circuit_type in "${circuit_type_list[@]}"; do
    for circuit_size in $(seq 12 12); do
        echo "Processing: ${circuit_type}_${circuit_size}"
        folder="./out/${circuit_type}_${circuit_size}"
        cutqc="$folder/cutqc.txt"
        mkdir -p "$(dirname "$cutqc")"
        
        mlft="$folder/mlft.txt"
        mkdir -p "$(dirname "$mlft")"

        figs_folder="$folder/figs"
        mkdir $figs_folder

        for iteration in $(seq 1 2); do
            exec 3>>$cutqc
            exec 4>>$mlft

            # Flag to control output switching
            use_output1=true

            # Run the Python script and process its output line by line
            python full_cut_mlft.py $circuit_type $circuit_size $folder | while IFS= read -r line; do
                if [[ "$line" == "Start MLFT" ]]; then
                    use_output1=false  # Change flag to switch output file
                    continue  # Skip the line with "Start MLFT"
                fi

                if $use_output1; then
                    echo "$line" >&3
                else
                    echo "$line" >&4
                fi
            done
        done
        # Close file descriptors
        exec 3>&-
        exec 4>&-
    done
done

echo "Batch processing completed."
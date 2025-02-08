#!/bin/bash

# Define the folder containing model configuration files
config_folder="configuration/Disaster"

# Define the log file to store the status
log_file="model_run_status.log"

# Clear the log file at the beginning
> "$log_file"

# Loop through each JSON file in the specified folder
for model_config in "$config_folder"/*.json; do
    echo "Running main.py with configuration $model_config..."
    
    # Run the command for each model and log the output to the log file
    python main.py --config "$model_config" --model.pretrained --training.epochs 25 --training.num_workers 8

    # Check if the command ran successfully and log the result
    if [ $? -eq 0 ]; then
        echo "$(date) - Successfully completed: $model_config" >> "$log_file"
    else
        echo "$(date) - Error occurred while running: $model_config" >> "$log_file"
    fi
done

echo "Model run status has been logged to $log_file."

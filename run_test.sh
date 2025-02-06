#!/bin/bash

# Define the folder containing model configuration files
config_folder="configuration/Disaster"

# Define the log file to store the status
log_file="model_run_status.log"

# Clear the log file at the beginning
> "$log_file"

# Loop through each JSON file in the specified folder
for model_config in "$config_folder"/*.json; do

    project_dir=$(grep -oP '"project_dir":\s*"\K[^"]+' $model_config)
    model_name=$(grep -oP '"name":\s*"\K[^"]+' $model_config)
    echo "Running main.py with configuration $model_config..."
    
    # echo $model_name, $project_dir, path = Logs/$project_dir/$model_name"_val_best.pth"
    # # Run the command for each model and log the output to the log file
    python inference.py --config "$model_config" --model.weight_path Logs/$project_dir/$model_name"_val_best.pth"

    # Check if the command ran successfully and log the result
    if [ $? -eq 0 ]; then
        echo "$(date) - Successfully completed: $model_config" >> "$log_file"
    else
        echo "$(date) - Error occurred while running: $model_config" >> "$log_file"
    fi
done

echo "Model run status has been logged to $log_file."

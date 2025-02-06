#!/bin/bash

# Delete all directories named "checkpoint"
echo "Deleting all directories named 'checkpoint'..."
find . -type d -name "checkpoint" -exec rm -rf {} +
echo "All 'checkpoint' directories deleted."

# Delete all files named "after_train.pth"
echo "Deleting all files named 'after_train.pth'..."
find . -type f -name "*after_train.pth" -exec rm -f {} +
echo "All '*after_train.pth' files deleted."

echo "Cleanup completed!"

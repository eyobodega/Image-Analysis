import shutil
import os

def delete_folders(folders_list):
    for folder_path in folders_list:
        try:
            # Check if the folder exists
            if os.path.exists(folder_path):
                # Remove the folder
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")
            else:
                print(f"Folder not found: {folder_path}")
        except Exception as e:
            print(f"An error occurred while deleting folder {folder_path}: {e}")

# List of folders to delete
folders_to_delete = [
    'staples_dataset/images/train',
    'staples_dataset/images/val',
    'staples_dataset/train_labels',
    'staples_dataset/val_labels'
]

# Call the function to delete folders
delete_folders(folders_to_delete)

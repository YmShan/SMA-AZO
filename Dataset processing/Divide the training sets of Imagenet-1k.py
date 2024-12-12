import tarfile
import os

# Specify the folder path containing .tar files
folder_path = "./train"

# Get all .tar files in the folder
tar_files = [file for file in os.listdir(folder_path) if file.endswith(".tar")]

# Iterate through each .tar file
for tar_file in tar_files:
    # Create the corresponding folder
    folder_name = tar_file[:-4]  # Remove the extension
    os.makedirs(os.path.join(folder_path, folder_name), exist_ok=True)
    
    # Extract files to the corresponding folder
    with tarfile.open(os.path.join(folder_path, tar_file), "r") as tar:
        tar.extractall(os.path.join(folder_path, folder_name))


import os

def delete_tar_files(directory):
    # Iterate through all files and subdirectories in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file ends with .tar
            if file.endswith(".tar"):
                file_path = os.path.join(root, file)
                # Delete the file
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

# Specify the directory path where .tar files should be deleted
directory_path = "./train"

# Call the function to delete all .tar files in the directory
delete_tar_files(directory_path)

print('Done')

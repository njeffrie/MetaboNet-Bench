import os
import zipfile
import glob

def extract_all_zipfiles():
    """
    Extract all zip files in the current directory to directories named after the zip file.
    """
    # Get current directory
    current_dir = os.getcwd()
    print(f"Extracting zip files in: {current_dir}")
    
    # Find all zip files in current directory
    zip_files = glob.glob("*.zip")
    
    print(f"Found {len(zip_files)} zip files: {zip_files}")
    
    for zip_file in zip_files:
        # Get the directory name (zip filename without .zip extension)
        dir_name = os.path.splitext(zip_file)[0]
        
        print(f"\nExtracting {zip_file} to {dir_name}/")
        
        # Create directory if it doesn't exist
        os.makedirs(dir_name, exist_ok=True)
        
        try:
            # Extract the zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(dir_name)
            print(f"✓ Successfully extracted {zip_file}")
            
        except zipfile.BadZipFile:
            print(f"✗ Error: {zip_file} is not a valid zip file or is corrupted")
        except Exception as e:
            print(f"✗ Error extracting {zip_file}: {str(e)}")
    
    print("\nExtraction complete!")

if __name__ == "__main__":
    extract_all_zipfiles()

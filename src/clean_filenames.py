import os
import re

def clean_filename(filename):
    # First replace +- with space
    cleaned = filename.replace('+-', ' ')
    # Then replace any remaining + with space
    cleaned = cleaned.replace('+', ' ')
    # Remove any extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Keep only alphanumeric characters, spaces, hyphens, underscores, and dots
    cleaned = "".join([c for c in cleaned if c.isalnum() or c in (' ', '-', '_', '.')]).rstrip()
    return cleaned

def process_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist")
        return
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            old_path = os.path.join(directory_path, filename)
            new_filename = clean_filename(filename)
            new_path = os.path.join(directory_path, new_filename)
            
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"Error renaming {filename}: {str(e)}")

def main():
    directories = [
        "journal_downloads/The_Indian_Journal_of_Agricultural_Sciences",
        "journal_downloads/The_Indian_Journal_of_Animal_Sciences"
    ]
    
    for directory in directories:
        print(f"\nProcessing directory: {directory}")
        process_directory(directory)

if __name__ == "__main__":
    main() 
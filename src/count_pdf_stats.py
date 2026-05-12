import os
import fitz  # PyMuPDF
from tqdm import tqdm
from collections import defaultdict

def count_pages_in_pdf(pdf_path):
    """Count number of pages in a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return 0

def process_directories(paths):
    """Process multiple directories and count PDFs and pages."""
    # Dictionary to store stats per directory
    stats = defaultdict(lambda: {'files': 0, 'pages': 0})
    total_stats = {'files': 0, 'pages': 0}
    
    for path in paths:
        print(f"\nProcessing directory: {path}")
        pdf_files = [f for f in os.listdir(path) if f.lower().endswith('.pdf')]
        
        for pdf_file in tqdm(pdf_files, desc="Counting pages"):
            pdf_path = os.path.join(path, pdf_file)
            pages = count_pages_in_pdf(pdf_path)
            
            # Update directory stats
            stats[path]['files'] += 1
            stats[path]['pages'] += pages
            
            # Update total stats
            total_stats['files'] += 1
            total_stats['pages'] += pages
    
    return stats, total_stats

def print_stats(stats, total_stats):
    """Print statistics in a formatted way."""
    print("\n=== Directory Statistics ===")
    print("-" * 80)
    print(f"{'Directory':<60} {'Files':<10} {'Pages':<10}")
    print("-" * 80)
    
    for directory, stat in stats.items():
        # Get the last part of the directory path for display
        dir_name = os.path.basename(directory)
        print(f"{dir_name:<60} {stat['files']:<10} {stat['pages']:<10}")
    
    print("-" * 80)
    print(f"{'TOTAL':<60} {total_stats['files']:<10} {total_stats['pages']:<10}")
    print("-" * 80)

def main():
    # List of directories to process
    paths = [
        '/Users/ganeshnallagachu/Desktop/world_bank/downloads/Hindi/Farming',
        '/Users/ganeshnallagachu/Desktop/world_bank/downloads/Hindi/fruit and flower',
        "/Users/ganeshnallagachu/Desktop/world_bank/downloads/Traditional_Knowledge/Traditional Knowledge In Agriculture",
        "/Users/ganeshnallagachu/Desktop/world_bank/journal_downloads/Potato_Journal",
        "/Users/ganeshnallagachu/Desktop/world_bank/journal_downloads/The_Indian_Journal_of_Agricultural_Sciences",
        "/Users/ganeshnallagachu/Desktop/world_bank/journal_downloads/The_Indian_Journal_of_Animal_Sciences"
    ]
    
    # Process directories and get statistics
    stats, total_stats = process_directories(paths)
    
    # Print the results
    print_stats(stats, total_stats)

if __name__ == "__main__":
    main() 
import os
import re
from pathlib import Path


def create_replacer(replacement_text):
    """Creates a replacement function with the desired text"""
    def preserve_case_replace(match):
        original = match.group(0)
        # Simple replacement - you can add case preservation logic here if needed
        return replacement_text
    return preserve_case_replace


def replace_in_file(filepath, pattern, replacer, dry_run=True):
    """Replaces occurrences in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
        except:
            print(f"Unable to read: {filepath}")
            return 0
    except:
        print(f"Error reading: {filepath}")
        return 0
    
    # Count occurrences
    matches = pattern.findall(content)
    if not matches:
        return 0
    
    print(f"✓ {filepath}: {len(matches)} occurrence(s) found")
    
    if not dry_run:
        # Replace
        new_content = pattern.sub(replacer, content)
        
        # Write modified file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  → File modified")
        except:
            print(f"  Error writing file")
    
    return len(matches)


def find_and_replace_content(folder_path, search_text, replace_text, case_sensitive=False):
    """
    Searches and replaces text in all files within a folder.
    
    Args:
        folder_path: Path to the folder to process
        search_text: Text to search for
        replace_text: Replacement text
        case_sensitive: If False, search is case-insensitive
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist!")
        return
    
    # Search pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(search_text), flags)
    
    # Create replacement function
    replacer = create_replacer(replace_text)
    
    # Dry-run first
    print("\n" + "="*60)
    print(f"Search: '{search_text}' → '{replace_text}'")
    print("PREVIEW MODE - No changes will be made")
    print("="*60 + "\n")
    
    total_occurrences = 0
    files_affected = 0
    
    # Walk through all files
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            count = replace_in_file(filepath, pattern, replacer, dry_run=True)
            if count > 0:
                total_occurrences += count
                files_affected += 1
    
    print("\n" + "="*60)
    print(f"SUMMARY: {total_occurrences} occurrence(s) in {files_affected} file(s)")
    print(f"Changes: {search_text} --> {replace_text}")
    print("="*60 + "\n")
    
    if total_occurrences == 0:
        print("No occurrences found.")
        return
    
    # Ask for confirmation
    response = input("Do you want to perform the replacements? (yes/no): ").strip().lower()
    
    if response in ['oui', 'o', 'yes', 'y']:
        print("\n" + "="*60)
        print("REPLACEMENT IN PROGRESS...")
        print("="*60 + "\n")
        
        total_replaced = 0
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                filepath = os.path.join(root, filename)
                count = replace_in_file(filepath, pattern, replacer, dry_run=False)
                total_replaced += count
        
        print("\n" + "="*60)
        print(f"COMPLETED: {total_replaced} occurrence(s) replaced")
        print("="*60 + "\n")
    else:
        print("Operation cancelled.\n")


def get_new_name(old_name, pattern, replacement):
    """Replaces the pattern in the name (case-insensitive)"""
    return pattern.sub(replacement, old_name)


def find_and_rename_items(folder_path, search_text, replace_text, case_sensitive=False):
    """
    Finds and renames files and folders containing the pattern.
    
    Args:
        folder_path: Path to the folder to process
        search_text: Text to search for in names
        replace_text: Replacement text
        case_sensitive: If False, search is case-insensitive
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(search_text), flags)
    
    items_to_rename = []
    
    # Walk through all files and folders (bottom-up for folders)
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Process files
        for filename in files:
            if pattern.search(filename):
                old_path = os.path.join(root, filename)
                new_filename = get_new_name(filename, pattern, replace_text)
                new_path = os.path.join(root, new_filename)
                items_to_rename.append((old_path, new_path, 'file'))
        
        # Process folders
        for dirname in dirs:
            if pattern.search(dirname):
                old_path = os.path.join(root, dirname)
                new_dirname = get_new_name(dirname, pattern, replace_text)
                new_path = os.path.join(root, new_dirname)
                items_to_rename.append((old_path, new_path, 'folder'))
    
    if not items_to_rename:
        print(f"\n✓ No files or folders to rename found for pattern '{search_text}'.\n")
        return 0
    
    # Display planned renames
    print(f"\n{'='*80}")
    print(f"Rename: '{search_text}' → '{replace_text}'")
    print(f"{'Type':<10} {'Old name':<35} → {'New name':<35}")
    print(f"{'='*80}")
    
    for old_path, new_path, item_type in items_to_rename:
        old_name = os.path.basename(old_path)
        new_name = os.path.basename(new_path)
        rel_dir = os.path.relpath(os.path.dirname(old_path), folder_path)
        if rel_dir == '.':
            rel_dir = '(root)'
        
        print(f"{item_type:<10} {old_name:<35} → {new_name:<35}")
        print(f"           In: {rel_dir}")
    
    print(f"{'='*80}\n")
    print(f"TOTAL: {len(items_to_rename)} item(s) to rename\n")
    
    # Ask for confirmation
    response = input("Do you want to perform the renaming? (yes/no): ").strip().lower()
    
    if response in ['oui', 'o', 'yes', 'y']:
        print("\n" + "="*80)
        print("RENAMING IN PROGRESS...")
        print("="*80 + "\n")
        
        # Perform renames
        success_count = 0
        for old_path, new_path, item_type in items_to_rename:
            try:
                # Check if new name already exists
                if os.path.exists(new_path):
                    print(f"⚠️  WARNING: {new_path} already exists, rename skipped")
                    continue
                
                os.rename(old_path, new_path)
                print(f"✓ Renamed: {os.path.basename(old_path)} → {os.path.basename(new_path)}")
                success_count += 1
            except Exception as e:
                print(f"Error renaming {old_path}: {e}")
        
        print("\n" + "="*80)
        print(f"COMPLETED: {success_count} item(s) renamed")
        print("="*80 + "\n")
        
        return success_count
    else:
        print("Operation cancelled.\n")
        return 0


def main():
    """Main function that performs multiple find/replace operations"""
    folder_path = input("Enter the folder path to process: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist!")
        return
    
    # List of content replacements to perform
    # Format: (search_text, replacement_text)
    content_replacements = [
       ("PolyMAC_P0P1NC", "ADEFZAFAAFRGZGAFA_HFV"),
       ("PolyMAC_P0", "ADEFZAFAAFRGZGAFA_MPFA"),
       ("Option_PolyMAC", "Option_ADEFZAFAAFRGZGAFA_family"),
       ("Domaine_Cl_PolyMAC", "Domaine_Cl_ADEFZAFAAFRGZGAFA_family"),
       ("is_polymac_family", "is_poly_family"),
       ("PolyMAC", "PolyMAC_CDO"),
       ("ADEFZAFAAFRGZGAFA", "PolyMAC")]
    
    # List of file/folder name replacements to perform
    # Format: (search_text, replacement_text)
    name_replacements = [
       ("PolyMAC_P0P1NC", "ADEFZAFAAFRGZGAFA_HFV"),
       ("PolyMAC_P0", "ADEFZAFAAFRGZGAFA_MPFA"),
       ("Option_PolyMAC", "Option_ADEFZAFAAFRGZGAFA_family"),
       ("Domaine_Cl_PolyMAC", "Domaine_Cl_ADEFZAFAAFRGZGAFA_family"),
       ("PolyMAC", "PolyMAC_CDO"),
       ("ADEFZAFAAFRGZGAFA", "PolyMAC")] 
    
    print(f"\n{'='*80}")
    print(f"Target folder: {folder_path}")
    print(f"Content replacements to perform: {len(content_replacements)}")
    print(f"Name replacements to perform: {len(name_replacements)}")
    print(f"{'='*80}")
    
    # PHASE 1: Replace content in files
    if content_replacements:
        print("\n" + "#"*80)
        print("PHASE 1: REPLACING CONTENT IN FILES")
        print("#"*80)
        
        for i, (search, replace) in enumerate(content_replacements, 1):
            print(f"\n### CONTENT REPLACEMENT {i}/{len(content_replacements)} ###")
            find_and_replace_content(folder_path, search, replace, case_sensitive=False)
    
    # PHASE 2: Rename files and folders
    if name_replacements:
        print("\n" + "#"*80)
        print("PHASE 2: RENAMING FILES AND FOLDERS")
        print("#"*80)
        
        for i, (search, replace) in enumerate(name_replacements, 1):
            print(f"\n### NAME REPLACEMENT {i}/{len(name_replacements)} ###")
            find_and_rename_items(folder_path, search, replace, case_sensitive=False)
    
    print("\n" + "="*80)
    print("✅ ALL OPERATIONS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
    print("/!\ /!\ /!\ /!\ /!\ ")
    print("Warning, calls to debute_par(PolyMAC) have been changed to debute_par(PolyMAC_CDO), might need a revert to debute_par(PolyMAC)")
    print("Be careful regarding folder names: \n PolyMAC might have been previously used as the name of the PolyMAC_family and has been changed to PolyMAC_CDO")

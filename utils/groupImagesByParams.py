import os
import pandas as pd
import shutil
import re # Import regular expressions for sanitizing names

def sanitize_foldername(name):
    """
    Cleans a string to be suitable as a directory name.
    Replaces spaces with underscores and removes characters not allowed in filenames.
    Handles potential None or NaN values.
    """
    if pd.isna(name) or not name:
        return "Unknown_Artist" # Default name for missing artists
    
    # Convert to string
    name = str(name)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Remove characters that are problematic in directory names across OS
    name = re.sub(r'[\\/*?:"<>|]', '', name)
    # Remove leading/trailing whitespace or underscores
    name = name.strip('._ ') 
    
    # If the name becomes empty after sanitization, use a default
    if not name:
        return "Invalid_Artist_Name"
        
    return name

def group_images_by_artist(image_dir, csv_path, output_dir):
    """
    Groups artwork images into directories named after their artists based on a CSV file.

    Args:
        image_dir (str): Path to the directory containing the source image files.
        csv_path (str): Path to the CSV file containing artwork metadata.
                        Expected columns: 'artist', 'new_filename'.
        output_dir (str): Path to the directory where the artist folders with 
                          images will be created.
    """

    print(f"Reading artwork information from: {csv_path}")
    try:
        # Read the CSV data using pandas, skipping potentially empty lines
        df = pd.read_csv(csv_path, skip_blank_lines=True)
        print(f"Found {len(df)} records in the CSV file.")
        
        # Check for required columns
        if 'artist' not in df.columns or 'new_filename' not in df.columns:
             print(f"Error: Required columns ('artist', 'new_filename') not found in {csv_path}")
             return
             
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty or invalid: {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Create the main output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/checked at: {output_dir}")

    processed_count = 0
    skipped_missing_file = 0
    skipped_missing_info = 0

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        try:
            # Get artist name and filename, handle potential missing values
            artist_name = row['artist']
            image_filename = row['new_filename']

            if pd.isna(image_filename) or not str(image_filename).strip():
                print(f"Warning: Skipping row {index+2} due to missing filename.")
                skipped_missing_info += 1
                continue
                
            # Convert filename to string just in case
            image_filename = str(image_filename).strip()

            # Sanitize artist name for the directory name
            artist_foldername = sanitize_foldername(artist_name)

            # --- Handle Image File ---
            source_image_path = os.path.join(image_dir, image_filename)
            artist_dest_dir = os.path.join(output_dir, artist_foldername)
            dest_image_path = os.path.join(artist_dest_dir, image_filename)
            
            dest_info_path = os.path.join(artist_dest_dir, image_filename)

            # Create artist-specific directory if it doesn't exist
            os.makedirs(artist_dest_dir, exist_ok=True)

            # Check if the source image exists before copying
            if os.path.exists(source_image_path):
                # Copy the file
                shutil.copy2(source_image_path, dest_image_path) # copy2 preserves metadata
                processed_count += 1
            else:
                # Print a warning if the image file is not found in the source directory
                print(f"Warning: Source image not found: {source_image_path}. Skipping.")
                skipped_missing_file += 1
                # Optionally remove the created artist directory if it's now empty and you want clean output
                # try:
                #     if not os.listdir(artist_dest_dir):
                #         os.rmdir(artist_dest_dir)
                # except OSError: 
                #     pass # Ignore error if dir is not empty (another file might be there already)
                continue

            # Print progress update periodically
            if (processed_count + skipped_missing_file + skipped_missing_info) % 500 == 0:
                print(f"Processed {processed_count + skipped_missing_file + skipped_missing_info} records...")

        except Exception as e:
            # Catch any other errors during row processing
            print(f"Error processing row {index+2} (Filename: {row.get('new_filename', 'N/A')}): {e}")
            skipped_missing_info += 1

    print("\n--- Processing Summary ---")
    print(f"Total records analyzed in CSV: {len(df)}")
    print(f"Successfully copied: {processed_count} images.")
    print(f"Skipped due to missing source image file: {skipped_missing_file}")
    print(f"Skipped due to missing info in CSV or other errors: {skipped_missing_info}")
    print(f"Organized image files saved under: {output_dir}")
    print("--------------------------")


# ----- КАК ИСПОЛЬЗОВАТЬ -----
# 1. Замените пути-заполнители ниже на ваши реальные пути.
#    Убедитесь, что пути указаны правильно для вашей операционной системы (используйте / или \\).
# 2. Убедитесь, что у вас установлена библиотека 'pandas' (`pip install pandas`).
# 3. Запустите скрипт из вашего терминала: python имя_вашего_скрипта.py

# **Конфигурация:**
source_images_folder = r''  # <-- CHANGE THIS to your image folder path
metadata_csv_file = r'.csv' # <-- CHANGE THIS to your CSV file path
output_base_folder = r'\outp'      # <-- CHANGE THIS to where you want the organized folders

# **Запуск процесса организации**
group_images_by_artist(source_images_folder, metadata_csv_file, output_base_folder)

print("\nОбработка завершена.")
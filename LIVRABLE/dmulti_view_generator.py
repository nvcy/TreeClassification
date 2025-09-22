import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from pathlib import Path
import json
import sys
import os

# Set UTF-8 encoding for stdout to handle Unicode characters
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Directories
BASE_DIR = Path(__file__).parent.parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "outputs" / "multi_view" / "train"
OUTPUT_DIR_TEST = BASE_DIR / "outputs" / "multi_view" / "test"
PROGRESS_FILE = BASE_DIR / "multi_view_progress.json"
IMG_SIZE = 224  # standard CNN input size

def load_progress():
    """Load processing progress from file"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_progress(progress):
    """Save processing progress to file"""
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")

def is_already_processed(file_path, output_class_dir, n_views):
    """Check if a file has already been processed"""
    stem = file_path.stem
    expected_files = [output_class_dir / f"{stem}_view{i}.png" for i in range(n_views)]
    return all(f.exists() for f in expected_files)

def load_point_cloud(file_path: Path):
    """Load point cloud file as Nx3 or Nx6 array"""
    try:
        # Handle different encodings for text files
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        data = None
        
        for encoding in encodings:
            try:
                data = np.loadtxt(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if data is None:
            print(f"Could not decode {file_path} with any encoding")
            return None
            
        if data.shape[1] > 3:
            return data[:, :3]  # only XYZ
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_views(points: np.ndarray, n_views=6):
    """
    Generate multiple 2D views of the point cloud from different angles.
    Returns a list of numpy images.
    """
    views = []
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111, projection='3d')

    angles = np.linspace(0, 360, n_views, endpoint=False)
    for angle in angles:
        ax.clear()
        ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
        ax.view_init(elev=30, azim=angle)
        ax.set_axis_off()

        fig.canvas.draw()
        # Get RGBA buffer and convert to RGB
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]  # keep only RGB channels

        image = Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE))
        views.append(np.array(image))

    plt.close(fig)
    return views

def safe_print(message):
    """Safely print message handling Unicode characters"""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: replace problematic characters
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(safe_message)

def generate_dataset_views(data_dir: Path, output_dir: Path, n_views=5):
    """
    Generate multi-view images for all point clouds in a dataset with resume capability
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load progress
    progress = load_progress()
    processed_files = set(progress.get('processed_files', []))
    
    total_files = 0
    processed_count = len(processed_files)
    
    # Count total files first
    for class_path in data_dir.iterdir():
        if not class_path.is_dir():
            continue
        files = [f for f in class_path.iterdir() if f.suffix in (".pts", ".xyz", ".txt")]
        total_files += len(files)
    
    safe_print(f"Total files to process: {total_files}")
    safe_print(f"Already processed: {processed_count}")
    safe_print(f"Remaining: {total_files - processed_count}")

    for class_path in data_dir.iterdir():
        if not class_path.is_dir():
            continue

        output_class_dir = output_dir / class_path.name
        output_class_dir.mkdir(exist_ok=True)

        files = [f for f in class_path.iterdir() if f.suffix in (".pts", ".xyz", ".txt")]
        
        for f in files:
            # Create a unique identifier for this file
            file_id = str(f.relative_to(data_dir))
            
            # Skip if already processed
            if file_id in processed_files:
                safe_print(f"Skipping already processed: {file_id}")
                continue
            
            # Double-check by looking at output files
            if is_already_processed(f, output_class_dir, n_views):
                safe_print(f"Output files exist, skipping: {file_id}")
                processed_files.add(file_id)
                continue
            
            try:
                points = load_point_cloud(f)
                if points is None:
                    safe_print(f"Could not load point cloud: {file_id}")
                    continue

                views = create_views(points, n_views=n_views)
                
                # Save each view
                saved_views = 0
                for i, view in enumerate(views):
                    try:
                        view_name = output_class_dir / f"{f.stem}_view{i}.png"
                        Image.fromarray(view).save(view_name)
                        saved_views += 1
                    except Exception as e:
                        safe_print(f"Error saving view {i} for {file_id}: {e}")
                        break
                
                if saved_views == len(views):
                    # Mark as processed only if all views were saved successfully
                    processed_files.add(file_id)
                    processed_count += 1
                    safe_print(f"Generated {len(views)} views for {file_id} ({processed_count}/{total_files})")
                    
                    # Save progress every 10 files
                    if processed_count % 10 == 0:
                        progress['processed_files'] = list(processed_files)
                        save_progress(progress)
                        safe_print(f"Progress saved: {processed_count}/{total_files}")
                else:
                    safe_print(f"Failed to save all views for {file_id}")
                    
            except Exception as e:
                safe_print(f"Error processing {file_id}: {e}")
                continue
    
    # Final progress save
    progress['processed_files'] = list(processed_files)
    save_progress(progress)
    safe_print(f"Final progress saved: {len(processed_files)}/{total_files}")

def main():
    try:
        generate_dataset_views(TEST_DIR, OUTPUT_DIR_TEST)
        safe_print("Multi-view generation complete!")
    except KeyboardInterrupt:
        safe_print("\nProcessing interrupted by user. Progress has been saved.")
    except Exception as e:
        safe_print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
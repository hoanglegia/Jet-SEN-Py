import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import glob
import math

class FingerprintApp:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("--- Model loaded successfully! ---")

    def preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        img = cv2.resize(img, (90, 90))
        img = img.reshape(90, 90, 1).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def find_all_altered_versions(self, real_path):
        filename = os.path.basename(real_path)
        base_name_no_ext = os.path.splitext(filename)[0]
        project_root = os.path.dirname(os.path.dirname(real_path))

        search_pattern = f"{base_name_no_ext}_*.BMP"
        folders = [
            os.path.join(project_root, "Altered", "Altered-Easy"),
            os.path.join(project_root, "Altered", "Altered-Medium"),
            os.path.join(project_root, "Altered", "Altered-Hard")
        ]
        
        matched_files = []
        for folder in folders:
            if os.path.exists(folder):
                full_pattern = os.path.join(folder, search_pattern)
                matched_files.extend(glob.glob(full_pattern))
        return matched_files

    def show_all_results(self, real_path, altered_list, threshold=0.5):
        """Display all results on a grid."""
        num_altered = len(altered_list)
        total_plots = num_altered + 1 # +1 for the Real image
        
        # Calculate the number of rows and columns for the grid (example: maximum 4 columns)
        cols = 4
        rows = math.ceil(total_plots / cols)

        fig = plt.figure(figsize=(20, 5 * rows))
        plt.suptitle(f"Results for ID: {os.path.basename(real_path)}", fontsize=18, fontweight='bold')

        # 1. Display the original Real image in the first box.
        img_real_raw = cv2.imread(real_path)
        img_real_raw = cv2.cvtColor(img_real_raw, cv2.COLOR_BGR2RGB)
        img_real_input = self.preprocess(real_path)

        ax = fig.add_subplot(rows, cols, 1)
        ax.imshow(img_real_raw)
        ax.set_title("REFERENCE", color='blue', fontsize=12, fontweight='bold')
        ax.axis('off')

        # 2. Loop through each Altered image, predict, and display results.
        print(f"Performing inference for {num_altered} images...")
        for i, alt_path in enumerate(altered_list):
            img_alt_raw = cv2.imread(alt_path)
            img_alt_raw = cv2.cvtColor(img_alt_raw, cv2.COLOR_BGR2RGB)
            img_alt_input = self.preprocess(alt_path)

            # Predict
            prediction = self.model.predict([img_real_input, img_alt_input], verbose=0)
            score = prediction[0][0]
            is_match = score > threshold

            # Plot
            ax = fig.add_subplot(rows, cols, i + 2)
            ax.imshow(img_alt_raw)
            
            color = 'green' if is_match else 'red'
            status = "MATCH" if is_match else "NO MATCH"
            
            # Get the name of the alteration type (Easy/Med/Hard) from the path
            folder_name = alt_path.split(os.sep)[-2] 
            
            ax.set_title(f"{folder_name}\nScore: {score:.4f}\n[{status}]", color=color, fontsize=11)
            
            # Draw a colored border around the image
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        print("Completed! Displaying comparison table...")
        plt.show()

    def show_single_pair(self, img1_path, img2_path, threshold=0.5):
        """Keep the original functionality for comparing 2 images"""
        img1 = self.preprocess(img1_path)
        img2 = self.preprocess(img2_path)
        if img1 is None or img2 is None: return

        score = self.model.predict([img1, img2], verbose=0)[0][0]
        is_match = score > threshold

        raw1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
        raw2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(raw1); ax1.set_title("Ảnh 1"); ax1.axis('off')
        ax2.imshow(raw2); ax2.set_title("Ảnh 2"); ax2.axis('off')
        
        color = 'green' if is_match else 'red'
        plt.figtext(0.5, 0.05, f"Result: {'MATCH' if is_match else 'NO MATCH'} (Score: {score:.4f})", 
                    ha="center", fontsize=14, bbox=dict(facecolor=color, alpha=0.3))
        plt.show()

def main():
    MODEL_PATH = "result/fingerprint_siamese_model.h5"
    app = FingerprintApp(MODEL_PATH)

    while True:
        print("\n--- FINGERPRINT RECOGNITION SYSTEM ---")
        print("1. Single Pair Comparison")
        print("2. All Altered Images (Grid View)")
        print("3. Exit")
        choice = input("Please choose an option (1/2/3): ")

        if choice == '1':
            p1 = input("Path image 1: ").strip().replace('"', '')
            p2 = input("Path image 2: ").strip().replace('"', '')
            if os.path.exists(p1) and os.path.exists(p2):
                app.show_single_pair(p1, p2)
        elif choice == '2':
            p_real = input("Path real image: ").strip().replace('"', '')
            if os.path.exists(p_real):
                altered_list = app.find_all_altered_versions(p_real)
                if altered_list:
                    app.show_all_results(p_real, altered_list)
                else:
                    print("No altered versions found.")
        elif choice == '3': break

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import os
import glob
import argparse

# The size of the square to draw at the center of each cell remains configurable here
# maybe TODO: have this as a parameter?
SQUARE_SIZE = 7

def create_feature_image(image_path, output_dir):
    """
    Loads an image with red contours, finds the center of each contoured object,
    and creates a new binary image with a square at each center.
    """
    # Load the image with contours
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return

    # Convert to HSV color space to easily isolate the red color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the bright red color used in the contour script (#ff3b30)
    # only works for franzeg not fogbank
    lower_red = np.array([0, 150, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red2 = np.array([170, 150, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    red_mask = mask1 + mask2

    # Use Connected Components Analysis to label each separate contour
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)

    # Create a new, completely black image to draw our features on
    feature_image = np.zeros(img.shape[:2], dtype=np.uint8)

    # Loop through each detected object (label), starting from 1 (0 is the background)
    if num_labels > 1:
        for i in range(1, num_labels):
            # Get the center of mass (centroid)
            cx, cy = int(centroids[i][0]), int(centroids[i][1])

            # Calculate the top-left corner of the square
            half_size = SQUARE_SIZE // 2
            x1 = cx - half_size
            y1 = cy - half_size
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE

            # Draw a filled white square on our new image
            cv2.rectangle(feature_image, (x1, y1), (x2, y2), (255), thickness=cv2.FILLED)

    # Save the final feature image
    base_filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, base_filename)
    cv2.imwrite(output_path, feature_image)
    print(f"Processed {base_filename}, found {num_labels - 1} features.")


if __name__ == "__main__":
    # Set up argument parser 
    parser = argparse.ArgumentParser(description="Create binary feature images from images with red contours.")
    parser.add_argument("--input-folder", required=True, help="Folder containing the images with red contours.")
    parser.add_argument("--output-parent", required=True, help="Parent directory for the new output folder (e.g., '.').")
    parser.add_argument("--output-folder", required=True, help="Name of the new folder to save feature images in.")
    
    args = parser.parse_args()

    #  Use parsed arguments to define paths 
    input_dir = args.input_folder
    output_dir = os.path.join(args.output_parent, args.output_folder)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files in the input directory
    search_path = os.path.join(input_dir, "*.tif")
    image_files = glob.glob(search_path)
    
    if not image_files:
        print(f"Error: No .tif images found in the '{input_dir}' directory.")
        print("Please check the path and make sure the contour script has run.")
    else:
        print(f"Found {len(image_files)} images in '{input_dir}'. Starting feature creation...")
        for file_path in sorted(image_files):
            create_feature_image(file_path, output_dir)
        print(f"\nProcessing complete. Feature images saved in '{output_dir}'.")
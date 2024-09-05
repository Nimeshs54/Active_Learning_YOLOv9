import json
import os

def convert_to_yolo(json_path, output_dir, image_width, image_height):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    captures = data['captures']
    
    for capture in captures:
        filename = capture['filename']
        annotations = capture['annotations'][0]['values']
        
        # Get the base name of the image file and create the corresponding txt file name
        base_name = os.path.splitext(os.path.basename(filename))[0]
        txt_filename = base_name + '.txt'
        
        yolo_annotations = []
        for annotation in annotations:
            label_id = annotation['label_id']
            x_center = (annotation['x'] + annotation['width'] / 2) / image_width
            y_center = (annotation['y'] + annotation['height'] / 2) / image_height
            width = annotation['width'] / image_width
            height = annotation['height'] / image_height
            
            yolo_annotations.append(f"{label_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Write YOLO formatted annotations to the text file
        txt_filepath = os.path.join(output_dir, txt_filename)
        with open(txt_filepath, 'w') as f:
            f.write('\n'.join(yolo_annotations))

def process_directory(json_dir, output_dir, image_width, image_height):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all JSON files in the directory
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            convert_to_yolo(json_path, output_dir, image_width, image_height)


# Parameters
json_dir = 'dataset/TRUCK Data/anno'  # Update with the path to your JSON directory
output_dir = 'dataset/TRUCK Data/labels'  # Update with the path to the directory where you want to save YOLO txt files
image_width = 1280  # Update with the width of your images
image_height = 960  # Update with the height of your images

# Process the directory
process_directory(json_dir, output_dir, image_width, image_height)

from PIL import Image, ImageDraw, ImageFont

# Load the image
image_path = 'bottomview_1702994651.jpg'
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
width, height = image.size

# Read the annotation file
annotation_path = 'bottomview_1702994651.txt'

with open(annotation_path, 'r') as file:
    lines = file.readlines()

# Load a font
try:
    font = ImageFont.truetype("arial.ttf", 15)
except IOError:
    font = ImageFont.load_default()

# Parse and draw bounding boxes
for line in lines:
    parts = line.strip().split()
    label = parts[0]
    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
    
    # Convert normalized coordinates to pixel values
    x_center *= width
    y_center *= height
    bbox_width *= width
    bbox_height *= height
    
    # Calculate the coordinates of the top-left and bottom-right corners
    x_min = int(x_center - bbox_width / 2)
    y_min = int(y_center - bbox_height / 2)
    x_max = int(x_center + bbox_width / 2)
    y_max = int(y_center + bbox_height / 2)
    
    # Draw bounding box
    draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)
    
    # Draw the label
    draw.text((x_min, y_min - 10), label, fill='red', font=font)

# Display the image with bounding boxes and labels
# image.show()

# Or save the image with bounding boxes and labels
image.save('annotated_image.png')

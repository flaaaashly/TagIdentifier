import os
import glob
import xml.etree.ElementTree as ET



# Define class labels (adjust if needed)
classes = ["tag"]

# Ensure the labels folder exists
output_dir = "labels"
os.makedirs(output_dir, exist_ok=True)

# Get all XML annotation files
xml_files = glob.glob("sample_label.xml")
print(f"Found {len(xml_files)} XML files.")

if not xml_files:
    print("No XML files found! Check the directory path.")
    exit()

# Process each XML file
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    txt_filename = os.path.join(output_dir, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))

    # Get image dimensions
    size = root.find("size")
    if size is None:
        print(f"Skipping {xml_file}: No <size> tag found!")
        continue

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # Open TXT file for writing annotations
    with open(txt_filename, "w") as txt_file:
        found_object = False 

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in classes:
                continue

            class_id = classes.index(class_name)

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Normalize to YOLO format (0-1)
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            txt_file.write(f"{class_id} {x_center:.2f} {y_center:.2f} {bbox_width:.2f} {bbox_height:.2f}\n")
            found_object = True  

        if found_object:
            print(f"Created: {txt_filename}")
        else:
            print(f"No valid objects in {xml_file}, skipping.")

print("Conversion complete! YOLO labels are in the 'labels' folder.")
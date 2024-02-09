import os
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import shutil

def read_annotations_xml(annotation_file):
    '''
    This function reads the annotations from the CVAT XML file and returns a pandas dataframe.

    Args:
    annotation_file (str): The path to the annotation file.
    '''
    # Parse the annotation file
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    # Divide the xml File into list of images
    images = root.findall('image')

    # Create empty pandas dataframe
    df = pd.DataFrame(columns=['image', 'label', 'connection', 'tl', 'br', 'rotation', 'c1', 'c2', 'c3', 'c4', 'c1_scaled', 'c2_scaled', 'c3_scaled', 'c4_scaled'])

    # Loop through the images
    for image in images:
        # Get the image name
        image_name = image.attrib['name']

        # Get image size
        img_width = int(image.attrib['width'])
        img_height = int(image.attrib['height'])

        # Get the boxes
        boxes = image.findall('box')
        for box in boxes:
            label = box.attrib['label']
            xtl = float(box.attrib['xtl'])
            ytl = float(box.attrib['ytl'])
            xbr = float(box.attrib['xbr'])
            ybr = float(box.attrib['ybr'])
            rotation = float(box.attrib['rotation']) if 'rotation' in box.attrib else 0

            # Calculate corner points of the rotated rectangle around the center
            center_x = (xtl + xbr) / 2
            center_y = (ytl + ybr) / 2
            width = xbr - xtl
            height = ybr - ytl

            # get the corners of the rotated rectangle
            corners = cv2.boxPoints(((center_x, center_y), (width, height), rotation))

            # Scale the corners to the image size
            corners_scaled = corners / [img_width, img_height]

            # Get the correct class to which the Beschriftung belongs
            if label == 'Beschriftung':
                for classes in box.findall('attribute'):
                    if classes.text == 'true':
                        connection = classes.attrib['name']
            else:
                connection = 'None'

            # Append to the dataframe
            df = pd.concat([df if not df.empty else None, 
                            pd.DataFrame({'image' : image_name, 'label': [label], 'connection' : [connection], 'tl': [(xtl, ytl)], 'br': [(xbr, ybr)], 'rotation': [rotation], 
                                          'c1' : [(corners[0][0], corners[0][1])], 
                                          'c2' : [(corners[1][0], corners[1][1])], 
                                          'c3' : [(corners[2][0], corners[2][1])], 
                                          'c4' : [(corners[3][0], corners[3][1])],
                                          'c1_scaled' : [(corners_scaled[0][0], corners_scaled[0][1])],
                                          'c2_scaled' : [(corners_scaled[1][0], corners_scaled[1][1])],
                                          'c3_scaled' : [(corners_scaled[2][0], corners_scaled[2][1])],
                                          'c4_scaled' : [(corners_scaled[3][0], corners_scaled[3][1])]})])
            
    return df


def visualize_annotations(df, image_dir, output_dir, mode='angle'):    
    '''
    This function visualizes the annotations and saves the images to the output directory.

    Args:
    df (pandas.DataFrame): The dataframe containing the annotations.
    image_dir (str): The directory containing the images.
    output_dir (str): The directory to save the images.
    mode (str): The mode to visualize the annotations. Options: 'angle' or 'corners'.
    '''
    # Delete and create the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Loop through the images and visualize the annotations
    for image in tqdm(df['image'].unique()):
        # Read the image
        img = cv2.imread(os.path.join(image_dir, image))

        # Get the annotations for the image
        annotations = df[df['image'] == image]

        # Plot the image
        fig, ax = plt.subplots(1)
        ax.axis('off')
        ax.imshow(img)

        # Loop through the annotations
        for index, row in annotations.iterrows():
            # Get the coordinates
            xtl = row['tl'][0]
            ytl = row['tl'][1]
            xbr = row['br'][0]
            ybr = row['br'][1]
            angle = row['rotation']
            c1 = row['c1']
            c2 = row['c2']
            c3 = row['c3']
            c4 = row['c4']

            if mode == 'angle':
                # Calculate the center of the rectangle
                center_x = (xtl + xbr) / 2
                center_y = (ytl + ybr) / 2

                # Create the rectangle
                rect = Rectangle((xtl, ytl), xbr - xtl, ybr - ytl, linewidth=0.5, edgecolor='r', facecolor='none')

                # Create the rotation transformation
                rotation = Affine2D().rotate_deg_around(center_x, center_y, angle)

                # Apply the transformation
                rect.set_transform(rotation + ax.transData)

                # Add the rectangle to the plot and color it blue
                ax.add_patch(rect)

            if mode == 'corners':
                # Create a polygon from the corners
                polygon = plt.Polygon([c1, c2, c3, c4], linewidth=0.5, closed=True, fill=None, edgecolor='b')

                # Add the polygon to the plot
                ax.add_patch(polygon)

        # Save the image
        plt.savefig(os.path.join(output_dir, image), bbox_inches='tight', dpi=300)


def visualize_scaled_annotations(df, image_dir, output_dir):    
    '''
    This function visualizes the annotations scaled to 1.0x1.0 by rescaling the corner points to the image size.

    Args:
    df (pandas.DataFrame): The dataframe containing the annotations.
    image_dir (str): The directory containing the images.
    output_dir (str): The directory to save the images.
    '''
    # Delete and create the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Loop through the images and visualize the annotations
    for image in tqdm(df['image'].unique()):
        # Read the image
        img = cv2.imread(os.path.join(image_dir, image))

        # Get image size
        img_height = img.shape[0]
        img_width = img.shape[1]

        # Get the annotations for the image
        annotations = df[df['image'] == image]

        # Plot the image
        fig, ax = plt.subplots(1)
        ax.axis('off')
        ax.imshow(img)

        # Loop through the annotations
        for index, row in annotations.iterrows():
            # Get the scaled coordinates and resclae them to the image size
            c1 = (row['c1_scaled'][0] * img_width, row['c1_scaled'][1] * img_height)
            c2 = (row['c2_scaled'][0] * img_width, row['c2_scaled'][1] * img_height)
            c3 = (row['c3_scaled'][0] * img_width, row['c3_scaled'][1] * img_height)
            c4 = (row['c4_scaled'][0] * img_width, row['c4_scaled'][1] * img_height)

            # Create a polygon from the corners
            polygon = plt.Polygon([c1, c2, c3, c4], linewidth=0.5, closed=True, fill=None, edgecolor='g')

            # Add the polygon to the plot
            ax.add_patch(polygon)

        # Save the image
        plt.savefig(os.path.join(output_dir, image), bbox_inches='tight', dpi=300)


def export_od_annotations_to_yolo(df, output_dir, classes):
    '''
    This function exports the OD annotations to the YOLO OBB format.

    Args:
    df (pandas.DataFrame): The dataframe containing the annotations.
    output_dir (str): The directory to save the annotations.
    classes (dict): The classes and their corresponding label.
    '''
    # Delete and create the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Iterate over all images
    for image in df['image'].unique():
        # Get the image dataframe
        df_image = df[df['image'] == image]

        # Get the boxes
        boxes = df_image[df_image['label'] != 'Beschriftung']

        # Write the annotations to a file
        file_name = f'{image}.txt'
        with open(os.path.join(output_dir, file_name), 'w') as file:
            for index, box in boxes.iterrows():
                label = classes[box['label']]

                x1 = box['c1_scaled'][0]
                y1 = box['c1_scaled'][1]
                x2 = box['c2_scaled'][0]
                y2 = box['c2_scaled'][1]
                x3 = box['c3_scaled'][0]
                y3 = box['c3_scaled'][1]
                x4 = box['c4_scaled'][0]
                y4 = box['c4_scaled'][1]

                file.write(f'{label} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n')


def export_ocr_annotations_to_yolo(df, output_dir, connections):
    '''
    This function exports the OCR annotations to the YOLO OBB format.

    Args:
    df (pandas.DataFrame): The dataframe containing the annotations.
    output_dir (str): The directory to save the annotations.
    classes (dict): The classes and their corresponding label.
    '''
    # Delete and create the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Iterate over all images
    for image in df['image'].unique():
        # Get the image dataframe
        df_image = df[df['image'] == image]

        # Get the boxes
        boxes = df_image[df_image['label'] == 'Beschriftung']

        # Write the annotations to a file
        file_name = f'{image}.txt'
        with open(os.path.join(output_dir, file_name), 'w') as file:
            for index, box in boxes.iterrows():
                connection = connections[box['connection']]

                x1 = box['c1_scaled'][0]
                y1 = box['c1_scaled'][1]
                x2 = box['c2_scaled'][0]
                y2 = box['c2_scaled'][1]
                x3 = box['c3_scaled'][0]
                y3 = box['c3_scaled'][1]
                x4 = box['c4_scaled'][0]
                y4 = box['c4_scaled'][1]

                file.write(f'{connection} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n')
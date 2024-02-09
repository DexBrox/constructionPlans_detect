'''
The script reads the annotations from the CVAT XML file, does some preprocessing and saves them to a pandas dataframe. 
It then visualizes the annotations and saves the images to the output directory. 
The mode parameter can be set to 'corners' for visualizing a oriented rectangle or 'angle' for visualizing a Polygon around the 4 calculated corner points. 
The resulting images are the same.
The calculated scaled corner points are then exported to the YOLO OBB format.

@Author: Theodor Kapler
@Project: SMARTPlan
'''

from helper_functions import read_annotations_xml, visualize_annotations, visualize_scaled_annotations, export_od_annotations_to_yolo, export_ocr_annotations_to_yolo


if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------------
    # Read the annotations from the CVAT XML file
    # ---------------------------------------------------------------------------------------------------
    df = read_annotations_xml(annotation_file='data/annotations.xml')

    # ---------------------------------------------------------------------------------------------------
    # Visualize the annotations (OD and OCR togehter)
    # ---------------------------------------------------------------------------------------------------
    visualize_annotations(df, image_dir='data/images', output_dir='vis_angles', mode='angle')
    visualize_annotations(df, image_dir='data/images', output_dir='vis_corners', mode='corners')
    visualize_scaled_annotations(df, image_dir='data/images', output_dir='vis_scaled')

    # ---------------------------------------------------------------------------------------------------
    # Export the scaled coordinates of the non text Labels to YOLO OBB format
    # ---------------------------------------------------------------------------------------------------
    # Define the classes
    classes = {'Schraube': '0',
               'Waermedaemmung': '1'}
    
    export_od_annotations_to_yolo(df, output_dir='result_od', classes=classes)

    # ---------------------------------------------------------------------------------------------------
    # Export the scaled coordinates of the text Labels to YOLO OBB format
    # ---------------------------------------------------------------------------------------------------
    export_ocr_annotations_to_yolo(df, output_dir='result_ocr', connections=classes)

            

'''
The script reads the annotations from the CVAT XML file, does some preprocessing and saves them to a pandas dataframe. 
It then visualizes the annotations and saves the images to the output directory. 
The mode parameter can be set to 'corners' for visualizing a oriented rectangle or 'angle' for visualizing a Polygon around the 4 calculated corner points. 
The resulting images are the same.
The calculated scaled corner points are then exported to the YOLO OBB format.

@Author: Theodor Kapler
@Project: SMARTPlan
'''

from helper_functions import *


if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------------
    # Read the annotations from the CVAT XML file
    # ---------------------------------------------------------------------------------------------------
    df = read_annotations_xml(annotation_file='data/annotations.xml')

    # save dataframe as markdown table
    df.to_markdown('data/annotations.md')

    # ---------------------------------------------------------------------------------------------------
    # Visualize the annotations (OD and OCR togehter)
    # ---------------------------------------------------------------------------------------------------
    #visualize_annotations(df, image_dir='data/images', output_dir='vis_corners', mode='corners')
    #visualize_scaled_annotations(df, image_dir='data/images', output_dir='vis_scaled')

    # ---------------------------------------------------------------------------------------------------
    # Export the scaled coordinates of the non text Labels to YOLO OBB format
    # ---------------------------------------------------------------------------------------------------
    # Röwaplan Original Classes:
    classes = {'Rohbau': '0',
               'Kunststoff': '1',
               'Dämmung' : '2',
               'Dübel' : '3',
               'Schraube' : '4',
               'Alublech' : '5',
               'Stahlblech' : '6',
               'Beschriftung': '7',
               'Folie' : '8',
               'PfostenRiegel': '9',
               'Konsole' : '10',
               'Edelstahl' : '11',
               'Glas' : '12',
               'Dichtung' : '13',
               'Holz' : '14',
               'Systemprofil' : '15'}

    export_od_annotations_to_yolo(df, output_dir='labels_od', classes=classes)

    # ---------------------------------------------------------------------------------------------------
    # Export the scaled coordinates of the text Labels to YOLO OBB format
    # ---------------------------------------------------------------------------------------------------
    # Röwaplan Original Connections:
    connections = {'Schraube': '0',
               'Dübel': '1',
               'Dämmung' : '2',
               'Alublech' : '3',
               'Stahlblech' : '4',
               'Kunststoff' : '5',
               'PfostenRiegel' : '6',
               'Folie': '7',
               'Allgemein' : '8',
               'Konsole' : '9',
               'Glas' : '10',
               'Dichtung' : '11',
               'Systemprofil' : '12',
               'None' : '13'}
    
    export_ocr_annotations_to_yolo(df, output_dir='labels_ocr', connections=connections)

    # ---------------------------------------------------------------------------------------------------
    # Read the created YOLO OBB files and visualize the annotations for verification
    # ---------------------------------------------------------------------------------------------------
    #visualize_yolo_annotations(yolo_dir='labels_od', output_dir='visualizations_od', mode='od')
    #visualize_yolo_annotations(yolo_dir='labels_ocr', output_dir='visualizations_ocr', mode='ocr')

            

        

            

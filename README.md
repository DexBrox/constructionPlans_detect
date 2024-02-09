# cvat2yolo-conversion

The script reads the annotations from the CVAT XML file, does some preprocessing and saves them to a pandas dataframe. 
It then visualizes the annotations and saves the images to the output directory. 
The mode parameter can be set to 'corners' for visualizing a oriented rectangle or 'angle' for visualizing a Polygon around the 4 calculated corner points. 
The resulting images are the same.
The calculated scaled corner points are then exported to the YOLO OBB format.

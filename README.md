# Face mask detection
This is the simple source code for face mask detection. Based on the face extracted from CenterFace approach, then apply the simple convolution network for image classification.

## Referent
CenterFace(size of 7.3MB) is a practical anchor-free face detection and alignment method for edge devices.
Please check this repo for the lated update of face detection model.
https://github.com/Star-Clouds/CenterFace

## Usage
Create the training face for CNN model
'''
python face_record.py
'''

Training and testing the face mask detection with camera: modify the "training" flag to True or False to active the training or mask detection process
'''
python mask_classification.py
'''

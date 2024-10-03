# Avataar_assignment


Tasks:

1. Task1. This task is to write an executable code that takes the input scene and the text prompt
from the command line argument and outputs an image with a red mask on all pixels where
the object (denoted in the text prompt) was present.
(e.g. python run.py --image ./example.jpg --class "chair" --output
./generated.png)

2. Task2. The second task is to change the pose of the segmented object by the relative angles
given by the user. You can use a consistent direction as positive azimuth and polar angle
change and mention what you used.
(e.g. python run.py --image ./example.jpg --class "chair" --azimuth
+72 --polar +0 --output ./generated.png)
The generated image:
a. Should preserve the scene (background)
b. Should adhere to the relative angles given by the user

Please refer to the branches Task1 and Task2 to find the proper code and explanations

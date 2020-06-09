You can find two files here. Both perform model inference, but they differ in what face detector is used which in turns affects what is done in the preprocessing pipeline.
Here are the things to keep in mind:

- **MTCNN** is the leading face detector. The inference scripts that use MTCNN perfrom full preprocessing pipeline and are better at
detecting faces, especially from the distance. However, this solution is slightly slower (around 1 FPS for video) than the alternative:

- **ResSsd** uses a face detector based on SSD framework (Single Shot MultiBox Detector), using a reduced ResNet-10 model.
It does not extract facial landmarks, hence the face alignment phase is not possible. This script was created towards the end of the 
project and has not been tested as thoroughly as the MTCNN model.

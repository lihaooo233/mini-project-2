# Project2
Hello everyone! 
This is project 2 of EC601, and I will show you how to use it now.

# Instruction（how to use）

Open the file 'project2', and we could find a rar file 'sunflower.rar', I have already stored all my photos here. So you should unzip the file first or you can not test the accuracy of the modules.

I have trained 2 modules and when you execute the .py file 'project2.py' you could select which module you want to test... I will introduce those two modules latter.
 
 # Idea of this program
 
  I used 500 photos of sunflowers and 500 photos of roses as my training data(300 photos each class),validation data(100 photos each class) and test data(100 photos). In order to guarantee the effect of training, I randomize the order of input(training data) and validation data.
  
 
 
 
 
 # About Modules and Comparsion

There are two trained modules (mode1 and mode2). Mode1 used 3 convolution layers and 2 all-connectionlayers, and the accuracy which tested by 200 photos is 75%. And the mode2 used 5 convolution layersand 2 all-connection layers, and the accuracy is 76.2%. So we could see that with same training epoch(which is 30 and 600 photos for training), mode2 which has more parameters than mode1, spent a more time for training and have a better validation accuracy and test accuracy. But the improvementis still limited. I believe the main reason of that is the training photos are too typical and have littlechange may reduce the test accuracy. So I believe adding more photos with different types may improve the test accuracy better. 



 # Contact

So, please let me know if you face any problem when you use this program especially for TA and professor.

contact: lihaooo@bu.edu

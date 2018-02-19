###########################################################
Moments in Time                                                    
###########################################################

The Moments in Time Dataset
http://moments.csail.mit.edu/

Mathew Monfort, Bolei Zhou, Sarah Adel Bargal,
Alex Andonian, Tom Yan, Kandan Ramakrishnan, Lisa Brown,
Quanfu Fan, Dan Gutfruend, Carl Vondrick, Aude Oliva


###########################################################
Relevant files:
###########################################################
README.txt (this file)
license.txt (the license file, this must be included)
trainingSet.csv (the training set annotations)
validationSet.csv (the validation set annotations)
moments_categories.txt (class index order)
training/ (folder containing training set mp4 videos)
validation/ (folder containing validation set mp4 videos)


###########################################################
training/
###########################################################
The directory contains mp4 videos used for training.
For example, the video filename "blocking/getty-karate-video-id635808620_4.mp4" is located at "training/blocking/getty-karate-video-id635808620_4.mp4"


###########################################################
validation/
###########################################################
The directory contains mp4 videos used for validation.


###########################################################
moments_categories.txt 
###########################################################
Contains each class label and index used for training


###########################################################
trainingSet.csv and validationSet.csv
###########################################################
Comma-seperated csv containing the video list and annotated label for the traning and validation sets.  The files contains the following fields:

- filename:
Unique filename for each video.
- label:
Annotated label for each video
- annotation positive responses:
Number of workers that agree with annotated label
- annotation negative responses:
Number of workers that disagree with annotated label

###########################################################

All cached videos can also be manually downloaded from http://data.csail.mit.edu/soundnet/actions3/.
For example,
http://data.csail.mit.edu/soundnet/actions3/dancing/yt-5of6ffo6-Bw_205.mp4
for the training video "dancing/yt-5of6ffo6-Bw_205.mp4".

# cs5293sp21-project2

### Name: Rithsek Ngem

## How to run the program:
Python redact.py --input filename
Python unredacted.py

## function:
* make_features(): this function will create a list of features extracted from the dataset
* make_features_redacted(): this function will create a list of features extracted from redacted file
### How features extracted: 
I extracted feature by the length of the Name, word index and character index of the name, and if there is space in Name. 
## Assumption 
The assumption I made during writting the program: 
* Read a lot of files(dataset) and extract the features 
* Train and test the features (Train_X)
* Read the redacted file and extract the features
* Train and test the features from redacted file
* Use KNeighborsClassifier to predict the Name in redacted file from Train_X
## Bug occure while writing code and expect to occure when executing
* spacy.load("en_core_web_lg") is not working, thus I cannot load more than 60 files for trainning
* the prediction success rate is around 50%, thus when testing with file before redacted, I expect to get an error


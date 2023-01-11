# Facial Recognition System
This is a project on how to recognize human face using Deep Learning and identify the authorized person

# Requirements
1. Python 3.10.3
```
$ conda create --name facial-recog python=3.10.3
```
2. Install All the requirements
```
$ pip install -r requirements.txt
```

# Command to run the server
```
$ conda activate facial-recog
$ python app.py
```

# Usage
## Testing Webpage
1. Run the server
2. Open the interface by interface.html through any browser(double click it)
3. Click submit and wait for result

## API
1. Run the server
2. Send your image data or image file using POST Method to http://localhost:5000/recognition
3. The return message will be : 
    * No Face Detected : No Human Face is detected
    * Welcome {person_name} : Authenticated Successfully
    * No Authorized Personnel detected : The individual in not authorized

## Adding New Authorized Person
1. Add authorized individual image in the database folder and rename the image as the individual's name
FROM python:3

WORKDIR /usr/src/app

ADD cropped_images .
ADD train.py .
ADD test.py .
ADD helper_func.py .
ADD labelled_pkl .
ADD train_test.txt .
ADD model .
ADD weights .
ADD requirements.txt .
ADD script.sh .
#RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
#make the script executable
#RUN ["chmod", "+x", "./script.sh"]
#changed the command to run the script
#CMD ./script.sh 
#you can read more about commands in docker at https://docs.docker.com
#add the command instruction
CMD [ "python", "./train.py" ]
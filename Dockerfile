FROM python:3

# get all necessary packages
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN pip3 install flask-restful
RUN pip3 install dill

# add our project
ADD . /

# expose the port for the API
EXPOSE 80

# run the API
CMD [ "python", "/server.py" ]

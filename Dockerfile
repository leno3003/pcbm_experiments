FROM eidos-service.di.unito.it/eidos-base-pytorch:2.2.1

# Copy source files and make it owned by the group eidoslab
# and give write permission to the group
COPY src /src
RUN chmod 775 /src
RUN chown -R :1337 /src

RUN pip3 install pytorchcv
RUN pip3 install -U nltk 
WORKDIR /src

RUN [ "python", "-c", "import nltk; nltk.download('wordnet', download_dir='/nltk_data'); nltk.download('omw-1.4', download_dir='/nltk_data')" ]

ENTRYPOINT ["python3"]

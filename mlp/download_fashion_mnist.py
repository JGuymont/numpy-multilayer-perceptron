#!/usr/bin/env python3
"""
Created on Sat Aug 18 2018
@author: J. Guymont
"""
from google_drive_downloader import GoogleDriveDownloader

if __name__ == "__main__":

    MNIST_TRAIN_FILE_ID = '1z8vHjGLPIaFU4VY-QfYa8pajIrWSZj87'
    MNIST_TEST_FILE_ID = '1u7RDFVb-B2R5zy-rdVuJHRZAI2ZuzesN'

    TRAIN_DEST_PATH = './data/csv/train.csv'
    TEST_DEST_PATH = './data/csv/test.csv'

    GoogleDriveDownloader.download_file_from_google_drive(
        file_id=MNIST_TRAIN_FILE_ID, 
        dest_path=TRAIN_DEST_PATH, unzip=False)
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id=MNIST_TEST_FILE_ID, 
        dest_path=TEST_DEST_PATH, unzip=False)
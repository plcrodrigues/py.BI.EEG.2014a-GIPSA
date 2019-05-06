#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import mne
import numpy as np
from braininvaders2013 import download as dl
import os
import glob
import zipfile
import yaml
from scipy.io import loadmat
from distutils.dir_util import copy_tree
import shutil

BI2014a_URL = 'https://zenodo.org/record/2669187/files/'

class BrainInvaders2014a():
    '''

    '''

    def __init__(self):

        self.subject_list = list(range(1, 65 + 1))

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)

        sessions = {}
        session_name = 'session_1'
        sessions[session_name] = {}
        run_name = 'run_1'

        chnames = ['FP1',
                    'FP2',
                    'F3',
                    'AFz',
                    'F4',
                    'T7',
                    'Cz',
                    'T8',
                    'P7',
                    'P3',
                    'Pz',
                    'P4',
                    'P8',
                    'O1',
                    'Oz',
                    'O2',
                    'STI 014']
        chtypes = ['eeg'] * 16 + ['stim']               

        D = loadmat(file_path)['samples'].T
        S = D[1:17,:]
        stim = D[-1,:]
        X = np.concatenate([S, stim[None,:]])

        info = mne.create_info(ch_names=chnames, sfreq=512,
                               ch_types=chtypes, montage='standard_1020',
                               verbose=False)
        raw = mne.io.RawArray(data=X, info=info, verbose=False)

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        # check if has the .zip
        url = BI2014a_URL + 'subject_' + str(subject).zfill(2) + '.zip'
        path_zip = dl.data_path(url, 'BRAININVADERS2014')
        path_folder = path_zip.strip('subject_' + str(subject).zfill(2) + '.zip')

        # check if has to unzip
        if not(os.path.isdir(path_folder + 'subject{:d}/'.format(subject))):
            print('unzip', path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        subject_paths = []

        # filter the data regarding the experimental conditions
        subject_paths.append(path_folder + 'subject_' + str(subject).zfill(2) + '/training.mat')

        return subject_paths

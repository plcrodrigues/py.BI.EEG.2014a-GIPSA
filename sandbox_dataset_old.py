
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances, ERPCovariances
from tqdm import tqdm
import numpy as np
import mne
from scipy.io import loadmat

scr = {}

for subject in [1, 2, 3, 4, 5]:

    #file_path = './data/subject_' + str(subject).zfill(2) + '.mat'
    file_path = '/nethome/coelhorp/mne_data/MNE-braininvaders2014-data/record/2669495/files/subject_' + str(subject).zfill(2) + '.mat'

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

    # filter data and resample
    fmin = 1
    fmax = 24
    raw.filter(fmin, fmax, verbose=False)

    # detect the events and cut the signal into epochs
    events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
    event_id = {'NonTarget': 1, 'Target': 2}
    epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=1.0, baseline=None, verbose=False, preload=True)
    epochs.pick_types(eeg=True)

    # get trials and labels
    X = epochs.get_data()
    y = events[:, -1]
    y = LabelEncoder().fit_transform(y)

    # cross validation
    skf = StratifiedKFold(n_splits=5)
    clf = make_pipeline(XdawnCovariances(estimator='lwf', classes=[1]), MDM())
    scr[subject] = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc').mean()

    # print results of classification
    print('subject', subject)
    print('mean AUC :', scr[subject])



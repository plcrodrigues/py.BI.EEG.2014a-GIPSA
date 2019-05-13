
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances, ERPCovariances
from tqdm import tqdm
from braininvaders2014a.dataset import BrainInvaders2014a
from scipy.io import loadmat
import numpy as np
import mne

from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
def score_func(y, y_pred):
    idx = (y == 1)
    return y_pred[idx].sum() / len(y_pred[idx])

scr = {}

dataset = BrainInvaders2014a()

for subject in dataset.subject_list:

    #load data
    print(subject)
    sessions = dataset._get_single_subject_data(subject)
    raw = sessions['session_1']['run_1']

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
    y = epochs.events[:, -1]
    y = LabelEncoder().fit_transform(y)

    # cross validation
    skf = StratifiedKFold(n_splits=5)
    clf = make_pipeline(XdawnCovariances(estimator='lwf', classes=[1]), MDM())
    scr[subject] = cross_val_score(clf, X, y, cv=skf, scoring = make_scorer(score_func)).mean()

    # print results of classification
    print('subject', subject)
    print('mean AUC :', scr[subject])

filename = './scores_p300.pkl'
joblib.dump(scr, filename)




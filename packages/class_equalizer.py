import os
import glob
import random as rnd
from sklearn.model_selection import train_test_split


class ClassEqualizer(object):
    def __init__(self):
        pass

    @staticmethod
    def get_splitted_and_classnormed_filelist(basepath='../datasets/course_dataset/', val_perc=0.3):

        # get the subdirectories in basepath folder
        subdirs = [subdir for subdir in os.walk(basepath)]
        if len(subdirs) == 0:
            raise FileExistsError('Could not find any subfolders in basepath (with classes!).')

        # extract the list of class subfolders, then walk over them and collect image paths
        subdirs = subdirs[0][1]
        class_files = {}
        for subdir in subdirs:
            for filename in glob.glob(os.path.join(basepath, subdir, '**', '*.png'), recursive=True):
                if subdir not in class_files:
                    class_files[subdir] = []

                class_files[subdir].append(filename)

        # now split the list into training and validation
        splitted = {'train': {}, 'test': {}}
        for classname in class_files:
            train, test = train_test_split(class_files[classname], test_size=val_perc)

            if classname not in splitted['train']:
                splitted['train'][classname] = []

            if classname not in splitted['test']:
                splitted['test'][classname] = []

            splitted['train'][classname].extend(train)
            splitted['test'][classname].extend(test)

        # random sample the class occurrences
        for sets in splitted:
            max_entries_key = sorted([(k, len(splitted[sets][k])) for k in splitted[sets]],
                                     key=lambda x: x[1],
                                     reverse=True)[0][0]

            for class_name in splitted[sets]:
                if class_name == max_entries_key:
                    continue

                # how many elements are missing ? (k)
                k = len(splitted[sets][max_entries_key]) - len(splitted[sets][class_name])

                # random sample then into the list
                splitted[sets][class_name].extend(rnd.sample(splitted[sets][class_name], k=k))

        return splitted['train'], splitted['test']




import os
import pickle


class CompetitionClassifierLoader(object):

    def __init__(self, file):
        """
        This class will load the Competition Save file.
        It reads:
        - FeatureExtractor object with respective parameters
        - The trained classifier with best performance on validation set.
        - The FeatureScaler object.
        - The LabelEncoder object.

        :param file: Competitions file path.
        """
        if os.path.isfile(file) is False:
            raise FileNotFoundError("Could not find" + str(file))

        competition_dict = pickle.load(open(file, 'rb'))
        results_key = 'competition'

        feature_extactor_key = 'extractor'
        label_encoder_key = 'label_encoder'
        scaler_key = 'scaler'

        best_accuracy = None
        best_acc_clsf = None
        best_clsf_nam = None

        feature_extractor = None
        labelEncoder = None
        scaler = None

        for permutations in competition_dict:
            if results_key not in permutations:
                raise KeyError("Could not find key " + str(results_key))

            for clsf_name in permutations[results_key]:
                score = permutations[results_key][clsf_name]
                if best_accuracy is None or score[1] > best_accuracy:
                    best_acc_clsf = score[0]
                    best_accuracy = score[1]
                    best_clsf_nam = clsf_name
                    feature_extractor = permutations[feature_extactor_key]
                    labelEncoder = permutations[label_encoder_key]
                    scaler = permutations[scaler_key]

        if best_acc_clsf is None:
            raise TypeError("Could not find any Classifier in Competitions File.")

        print("Loaded best classifier: " + str(best_clsf_nam) + " score: " + str(best_accuracy))

        self._clsf = best_acc_clsf[0]
        self._feature_extractor = feature_extractor
        self._label_encoder = labelEncoder
        self._scaler = scaler

    @property
    def scaler(self):
        return self._scaler

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def classifier(self):
        return self._clsf

    @property
    def feature_extractor(self):
        return self._feature_extractor


class GridClassifier(object):

    def __init__(self, classifier):
        self._classifier = classifier

    def classify_grid(self, grid_with_features):
        """
        This function returns a dictionary with
        all classified rectangles.
        The keys represent the classified class.
        :param grid_with_features:
        :return: dict({'class_a': [rects], 'class_b': [rects], ..})
        """
        class_predicted_rects = {}

        for shapes in grid_with_features:
            for rects in zip(grid_with_features[shapes][0],
                             grid_with_features[shapes][1]):

                if len(rects[1]) > 0:
                    prediction = self._classifier.predict([rects[1].ravel()]).ravel()[0]
                    if prediction not in class_predicted_rects:
                        class_predicted_rects[prediction] = []
                    class_predicted_rects[prediction].append(rects[0])

        return class_predicted_rects

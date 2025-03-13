import sys

import numpy as np
import quapy as qp
import pandas as pd
import quapy.functional as F

from quapy.protocol import APP, NPP
from freq_e import infer_freq_from_predictions
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats

# Classifiers
from sklearn.naive_bayes import GaussianNB as GaussianNBOriginal
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier as XGBClassifierOriginal

# Calibration
from quapy.classification.calibration import (
    LogisticCalibration as PlattScaling,
    BBQCalibration,
    IsotonicCalibration,
    HistogramCalibration,
    NBVSCalibration,
    BCTSCalibration,
    TSCalibration,
    VSCalibration,
)

# Quantification
from quapy.method.aggregative import (
    CC,
    ACC,
    PCC,
    PACC,
    MAX as TH_MAX,
    X as TH_X,
    T50 as TH_T50,
    MS as TH_MS,
    EMQ,
    HDy,
    newSVMKLD,
    newSVMAE,
    newSVMRAE,
)
from quapy.method.non_aggregative import DMx

# Metrics
from abstention.calibration import compute_ece as ece
from sklearn.metrics import (
    f1_score,
)
from quapy.error import (
    ae,
    se,
    rae,
)

qp.environ['SVMPERF_HOME'] = '/home/max/my_projects/QuaPy/svm_perf_quantification'


class XGBClassifier(XGBClassifierOriginal):
    def decision_function(self, X):
        assert len(self.classes_)==2, 'wrong number of classes'
        return self.predict_proba(X)[:,1]


class GaussianNB(GaussianNBOriginal):
    def decision_function(self, X):
        assert len(self.classes_)==2, 'wrong number of classes'
        return self.predict_proba(X)[:,1]


def str_to_class_or_func(name):
    return getattr(sys.modules[__name__], name)


def init_sets(config, random_state):
    populations = {}
    trainsets = {}
    testsets = {}
    for population_name, population_config in config['populations'].items():
        population_size = config['trainset']['total_size'] + config['testset']['total_size']
        population_config['n_samples'] = population_size
        X, y = make_classification(
            **population_config,
            weights=None,
            random_state=random_state,
        )
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        population = qp.data.LabelledCollection(X, y)
        trainset, testset = population.split_stratified(
            train_prop=config['trainset']['total_size']/population_size,
            random_state=random_state,
        )
        populations[population_name] = population
        trainsets[population_name] = trainset
        testsets[population_name] = testset
    return (trainsets, testsets)


def init_classifiers(config, n_jobs=None):

    classifiers = [None] * len(config['classification']['models'])

    for idx, model in enumerate(config['classification']['models']):
        classifier = {
            'model': model,
            'object': str_to_class_or_func(model)(),
            'calibration': 'No Calibration',
        }
        classifiers[idx] = classifier

    calibrated_classifiers = [{
        'model': classifier['model'],
        'calibration': calib_name,
        'object': str_to_class_or_func(calib_name)(
                classifier=str_to_class_or_func(classifier['model'])(),
                val_split=config['calibration']['val_split'],
                n_jobs=n_jobs,
            ),
        }
        for classifier in classifiers
        for calib_name in config['calibration']['methods']
        if calib_name != 'No Calibration'
    ]
    if 'No Calibration' in config['calibration']['methods']:
        classifiers.extend(calibrated_classifiers)
    else:
        classifiers = calibrated_classifiers

    return classifiers


def init_quantificators(config, classifiers, n_jobs):
    
    dont_calibrate = ('CC', 'ACC', 'TH_MAX', 'TH_X', 'TH_T50', 'TH_MS')

    quantificators = [{
        'model': classifier['model'],
        'calibration': classifier['calibration'],
        'quantifier_method': quantification_name,
        'object': str_to_class_or_func(quantification_name)(classifier['object']),
        }
        for classifier in classifiers
        for quantification_name in config['quantification']['methods']
        if (
            (quantification_name not in ('LR-Implicit', 'newSVMKLD', 'newSVMAE', 'newSVMRAE', 'HDx')) and
            (
                (quantification_name in dont_calibrate) and (classifier['calibration']=='No Calibration')
                or (quantification_name not in dont_calibrate)
            )
        )
    ]
    
    for quantificator in quantificators:
        if quantificator['quantifier_method'] in dont_calibrate:
            quantificator['calibration'] = '-'

    if 'LR-Implicit' in config['quantification']['methods']:
        lr_quantificators = [{
            'model': classifier['model'],
            'calibration': classifier.get('calibration') or 'No Calibration',
            'quantifier_method': 'LR-Implicit',
            'object': classifier['object'],
            }
            for classifier in classifiers
        ]
        quantificators.extend(lr_quantificators)
    SVM_methods = filter(lambda x: x.startswith('newSVM'), config['quantification']['methods'])
    for method in SVM_methods:
        qp_svm = str_to_class_or_func(method)(qp.environ['SVMPERF_HOME'])
        elm_quantificator = {
            'model': '-',
            'calibration': '-',
            'quantifier_method': f"ELM-{method.removeprefix('new')}",
            'object': qp_svm,
        }
        quantificators.extend([elm_quantificator])
    if 'HDx' in config['quantification']['methods']:
        qp_hdx = DMx.HDx(n_jobs=n_jobs)
        hdx_quantificator = {
            'model': '-',
            'calibration': '-',
            'quantifier_method': 'HDx',
            'object': qp_hdx,
        }
        quantificators.extend([hdx_quantificator])

    return quantificators


def make_predictions(
    quantif_method,
    quantificator_object,
    calibration,
    X_test,
    trainset_sample_prevalence,
):
    y_predict = None
    y_predict_proba = None
    if quantif_method != 'LR-Implicit':
        estimated_testset_prevalence = quantificator_object.quantify(X_test)
        if quantif_method not in ('newSVMKLD', 'newSVMAE', 'newSVMRAE', 'HDx'):
            y_predict = quantificator_object.classifier.predict(X_test)
            if calibration != '-':
                y_predict_proba = quantificator_object.classifier.predict_proba(X_test)
    else:
        y_predict = quantificator_object.predict(X_test)
        y_predict_proba = quantificator_object.predict_proba(X_test)
        freq_e_result = infer_freq_from_predictions(y_predict_proba[:, 1], trainset_sample_prevalence[1])
        estimated_testset_prevalence = np.asarray([1-freq_e_result['point'], freq_e_result['point']])

    return (estimated_testset_prevalence, y_predict, y_predict_proba)


def make_evaluations(
    quantification_config,
    classification_config,
    testset_sample,
    estimated_testset_prevalence,
    y_test,
    y_predict,
    y_predict_proba,
):
    eval_metrics = {}

    # Evaluate Quantification
    for eval_metric in quantification_config['metrics']:
        kwargs = {}
        if eval_metric in ['rae', 'nkld']:
            kwargs = {'eps': 1/(2*len(testset_sample))}
        eval_metrics[eval_metric] = str_to_class_or_func(eval_metric)(testset_sample.prevalence(), estimated_testset_prevalence, **kwargs)
    if y_predict is not None:
        # Evaluate Classification
        for eval_metric in classification_config['metrics']['labels']:
            if len(y_predict.shape) == 2:
                y_predict = np.argmax(y_predict, axis=1)
            if eval_metric == 'f1_score':
                kwargs = {'zero_division': np.nan}
            eval_metrics[eval_metric] = str_to_class_or_func(eval_metric)(y_test, y_predict, **kwargs)
    if y_predict_proba is not None:
        for eval_metric in classification_config['metrics']['scores']:
            if eval_metric == 'ece':
                eval_metrics[eval_metric] = str_to_class_or_func(eval_metric)(y_predict_proba, np.asarray([1-y_test, y_test]).T, bins=10)

    return eval_metrics


def run_sequence(config, n_jobs, random_state, with_eval=True):

    (trainsets, testsets) = init_sets(config, random_state)
    classifiers = init_classifiers(config, n_jobs)
    quantificators = init_quantificators(config, classifiers, n_jobs)

    n_rows = (
        len(config['populations']) *
        len(config['trainset']['sample_sizes']) *
        config['trainset']['n_prevalences'] *
        config['trainset']['repeats'] *
        len(quantificators) *
        len(config['testset']['sample_sizes']) *
        config['testset']['n_prevalences'] *
        config['testset']['repeats']
    )

    stats = pd.DataFrame(
        index=np.arange(n_rows),
        columns=[
            'Population',
            'Train sample size',
            'Train prev',
            'Test prev',
            'Test sample size',
            'Classifier',
            'Calibration',
            'Quantifier',
            'Estim test prev',
        ],
    )

    idx_row = 0

    for eval_metric in config['classification']['metrics']['labels']:
        stats[f'{eval_metric}'] = np.nan
    for eval_metric in config['classification']['metrics']['scores']:
        stats[f'{eval_metric}'] = np.nan
    for eval_metric in config['quantification']['metrics']:
        stats[f'{eval_metric}'] = np.nan

    for population_name, trainset in trainsets.items():
        print(f"Population: {population_name}")

        testset = testsets[population_name]
        for trainset_sample_size in config['trainset']['sample_sizes']:
            print(f"Train Sample Size: {trainset_sample_size}")
            
            print(f"Train Samples: {config['trainset']['n_prevalences']*config['trainset']['repeats']}")
            trainset_protocol = APP(
                data=trainset,
                sample_size=trainset_sample_size,
                n_prevalences=config['trainset']['n_prevalences'],
                repeats=config['trainset']['repeats'],
                smooth_limits_epsilon=0.01,
                random_state=random_state,
                return_type='labelled_collection',
            )

            for idx, trainset_sample in enumerate(trainset_protocol()):

                print(f"idx trainset_sample: {idx}")

                (X_train, y_train) = trainset_sample.Xy
                trainset_sample_prevalence = trainset_sample.prevalence()

                for quantificator in quantificators:

                    # Training Step

                    quantif_method = quantificator['quantifier_method']
                    quantificator_object = quantificator['object']
                    calibration = quantificator['calibration']
                    kwargs = {}
                    if quantif_method in config['quantification']['val_split'].keys():
                        kwargs = {'val_split': config['quantification']['val_split'][quantif_method]}

                    if quantif_method != 'LR-Implicit':
                        quantificator_object.fit(trainset_sample, **kwargs)
                    else:
                        quantificator_object.fit(X=X_train, y=y_train)

                    # Test Step

                    for testset_sample_size in config['testset']['sample_sizes']:

                        testset_protocol = APP(
                            data=testset,
                            sample_size=testset_sample_size,
                            n_prevalences=config['testset']['n_prevalences'],
                            repeats=config['testset']['repeats'],
                            smooth_limits_epsilon=0.01,
                            random_state=random_state,
                            return_type='labelled_collection',
                        )

                        for testset_sample in testset_protocol():
                            
                            qp.environ["SAMPLE_SIZE"] = testset_sample_size
                            
                            row_dict = {}
                            row_dict['Population'] = population_name
                            row_dict['Train sample size'] = trainset_sample_size
                            row_dict['Train prev'] = trainset_sample.prevalence()
                            row_dict['Classifier'] = quantificator['model']
                            row_dict['Calibration'] = quantificator.get('calibration')
                            row_dict['Quantifier'] = quantificator['quantifier_method']

                            row_dict['Test prev'] = testset_sample.prevalence()
                            row_dict['Test sample size'] = testset_sample_size

                            (X_test, y_test) = testset_sample.Xy

                            (estimated_testset_prevalence, y_predict, y_predict_proba) = make_predictions(
                                quantif_method,
                                quantificator_object,
                                calibration,
                                X_test,
                                trainset_sample_prevalence,
                            )
                            if np.isnan(estimated_testset_prevalence).any():
                                print("ERROR IN QUANTIFICATION")
                                print(row_dict)
                                stats.iloc[idx_row] = row_dict
                                idx_row += 1
                                continue
                            
                            row_dict['Estim test prev'] = estimated_testset_prevalence
                            
                            if with_eval:

                                # Eval quantifier (and classificator if possible)
                                eval_metrics = make_evaluations(
                                    config['quantification'],
                                    config['classification'],
                                    testset_sample,
                                    estimated_testset_prevalence,
                                    y_test,
                                    y_predict,
                                    y_predict_proba,
                                )
                                for eval_metric_name, eval_metric_value in eval_metrics.items():
                                    row_dict[f'{eval_metric_name}'] = eval_metric_value

                            stats.iloc[idx_row] = row_dict                
                            idx_row += 1


    return stats


def parse_stats(stats):
    stats['Train prev class_1'] = [x[1] for x in stats['Train prev']]
    stats['Test prev class_1'] = [x[1] for x in stats['Test prev']]
    stats['Abs prev shift'] = [round(abs(x[1]), 2) for x in (stats['Train prev'] - stats['Test prev'])]
    stats['Abs prev shift range'] = pd.cut(stats['Abs prev shift'], [-1, 0, 0.25, 0.5, 0.75, 1], labels=["[0]", "(0-0.25]", "(0.25-0.5]", "(0.5-0.75]", "(0.75-1.0]"])
    parsed_stats = stats.copy()
    cols_to_format = [
        'Train prev',
        'Test prev',
        'Estim test prev',
    ]
    for col in cols_to_format:
        parsed_stats[col] = parsed_stats[col].map(F.strprev)

    parsed_stats = parsed_stats.rename(columns={
        'rae': 'RAE',
        'ae': 'AE',
        'se': 'SE',
        'f1_score': 'F1',
        'ece': 'ECE',
        
        'Pupulation': 'Población',
        'Train sample size': 'Train n_samples',
        'Test sample size': 'Test n_samples',
        'Classifier': 'Clasificador',
        'Calibration': 'Calibración',
        'Quantifier': 'Cuantificador',
        'Train prev class_1': 'Train prev +',
        'Test prev class_1': 'Test prev +',
        'Abs prev shift': 'Dif prev abs',
        'Abs prev shift range': 'Abs Dataset Shift',
    })

    return parsed_stats


def create_ci_cols(group_cols, metric_cols, parsed_stats):
    summary = parsed_stats.groupby(group_cols)[metric_cols].agg(['mean', 'count', 'std'])
    summary = summary.rename(columns={'mean': 'media'})
    for metric in metric_cols:
        # Calculate the t-value for a 95% confidence interval
        t_value = stats.t.ppf(0.975, summary[(metric, 'count')] - 1)  # 0.975 corresponds to (1 - alpha/2)
        # Calculate the margin of error
        me = t_value * summary[(metric, 'std')] / (summary[(metric, 'count')] ** 0.5)
        # Calculate the lower and upper bounds of the confidence interval   
        summary[(metric, '(')] = summary[(metric, 'media')] - me
        summary[(metric, ')')] = summary[(metric, 'media')] + me
        summary = summary.drop((metric, 'count'), axis = 1)
        summary = summary.drop((metric, 'std'), axis = 1)
    return summary

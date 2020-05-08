from generators.csv_ import CSVGenerator
from model import efficientdet
from eval.common import evaluate_polyp
import os
import numpy as np

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    phi = 1
    weighted_bifpn = False
    common_args = {
        'batch_size': 1,
        'phi': phi,
    }
    test_generator = CSVGenerator(
        '../../polyp-datasets/CVC-VideoClinicDB-1.csv',
        '../../polyp-datasets/class_id.csv',
        shuffle_groups=False,
        **common_args
    )

    model_path = 'checkpoints/exp2/csv_16_0.2631_0.7740.h5'
    # model_path = '../checkpoints/exp1/csv_30_0.1828_0.7413.h5'

    input_shape = (test_generator.image_size, test_generator.image_size)
    anchors = test_generator.anchors
    num_classes = test_generator.num_classes()
    model, prediction_model = efficientdet(phi=phi, num_classes=num_classes, weighted_bifpn=weighted_bifpn)
    prediction_model.load_weights(model_path, by_name=True)
    average_precisions, recall, precision = evaluate_polyp(test_generator, prediction_model, visualize=False)

    # compute per class average precision
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations), test_generator.label_to_name(label),
              'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)
    mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
    print('mAP: {:.4f}'.format(mean_ap))


    mean_recall = np.average(recall)
    mean_precision = np.average(precision)

    if (mean_precision + mean_recall) <= 0:
        f_one = 0.0
        f_two = 0.0
    else:
        f_one = 2.0 * ((mean_precision * mean_recall) / (mean_precision + mean_recall))
        f_two = 5.0 * ((mean_precision * mean_recall) / ((4.0 * mean_precision) + mean_recall))

    print('mRecall: {:.4f}'.format(mean_recall))
    print('mPrecision: {:.4f}'.format(mean_precision))
    print('F1: {:.4f}'.format(f_one))
    print('F2: {:.4f}'.format(f_two))

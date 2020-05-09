"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import keras
from tensorflow import keras
import tensorflow as tf
from eval.common import evaluate
import numpy as np


class Evaluate(keras.callbacks.Callback):
    """
    Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.01,
        max_detections=100,
        save_path=None,
        tensorboard=None,
        weighted_average=False,
        verbose=1
    ):
        """
        Evaluate a given dataset using a given model at the end of every epoch during training.

        Args:
            generator: The generator that represents the dataset to evaluate.
            iou_threshold: The threshold used to consider when a detection is positive or negative.
            score_threshold: The score confidence threshold to use for detections.
            max_detections: The maximum number of detections to use per image.
            save_path: The path to save images with visualized detections to.
            tensorboard: Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average: Compute the mAP using the weighted average of precisions among classes.
            verbose: Set the verbosity level, by default this is set to 1.
        """
        self.generator = generator
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.weighted_average = weighted_average
        self.verbose = verbose
        self.active_model = model

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions, recall, precision = evaluate(
            self.generator,
            self.active_model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            visualize=False,
            more_metrics=True
        )

        # compute per class average precision
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)


        self.mean_recall = np.average(recall)
        self.mean_precision = np.average(precision)

        if ((self.mean_precision + self.mean_recall) <= 0):
            self.f_one = 0.0
            self.f_two = 0.0
        else:
            self.f_one = 2.0 * ((self.mean_precision * self.mean_recall) / (self.mean_precision + self.mean_recall))
            self.f_two = 5.0 * ((self.mean_precision * self.mean_recall) / ((4.0 *  self.mean_precision) + self.mean_recall))


        if self.tensorboard is not None:
            if tf.version.VERSION < '2.0.0' and self.tensorboard.writer is not None:
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = self.mean_ap
                summary_value.tag = "mAP"
                self.tensorboard.writer.add_summary(summary, epoch)

                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = self.mean_recall
                summary_value.tag = "mRecall"
                self.tensorboard.writer.add_summary(summary, epoch)

                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = self.mean_precision
                summary_value.tag = "mPrecision"
                self.tensorboard.writer.add_summary(summary, epoch)

                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = self.f_one
                summary_value.tag = "F1"
                self.tensorboard.writer.add_summary(summary, epoch)

                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = self.f_two
                summary_value.tag = "F2"
                self.tensorboard.writer.add_summary(summary, epoch)
            else:
                tf.summary.scalar('mAP', self.mean_ap, epoch)

        logs['mAP'] = self.mean_ap

        logs['mRecall'] = self.mean_recall
        logs['mPrecision'] = self.mean_precision
        logs['F1'] = self.f_one
        logs['F2'] = self.f_two

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))
            print('mRecall: {:.4f}'.format(self.mean_recall))
            print('mPrecision: {:.4f}'.format(self.mean_precision))
            print('F1: {:.4f}'.format(self.f_one))
            print('F2: {:.4f}'.format(self.f_two))

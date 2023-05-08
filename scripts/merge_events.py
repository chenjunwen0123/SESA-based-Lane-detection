import argparse
import os
import tensorflow as tf
from tensorflow.core.util import event_pb2
from tensorflow.core.framework import summary_pb2

def merge_events(event_file1, event_file2, output_event_file, start_step):
    writer = tf.summary.create_file_writer(output_event_file)

    # Write all records from event_file1 to output_event_file
    dataset1 = tf.data.TFRecordDataset(event_file1)
    with writer.as_default():
        for data in dataset1:
            event = event_pb2.Event.FromString(data.numpy())
            if event.HasField("summary"):
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        tf.summary.scalar(value.tag, value.simple_value, step=event.step)

    # Write records from event_file2 to output_event_file, starting from start_step
    dataset2 = tf.data.TFRecordDataset(event_file2)
    with writer.as_default():
        for data in dataset2:
            event = event_pb2.Event.FromString(data.numpy())
            if event.HasField("summary") and event.step >= start_step:
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        tf.summary.scalar(value.tag, value.simple_value, step=event.step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_event', type=str, required=True, help='Path to the first event file')
    parser.add_argument('--second_event', type=str, required=True, help='Path to the second event file')
    parser.add_argument('--joint_point', type=int, required=True, help='Joint step')
    args = parser.parse_args()

    first_event_file = args.first_event
    second_event_file = args.second_event
    start_step = args.joint_point
    merged_event_file = os.path.join(os.getcwd(), 'merged_event_file')

    merge_events(first_event_file, second_event_file, merged_event_file, start_step)

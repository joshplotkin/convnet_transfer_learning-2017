cd ~/tensorflow-github
touch WORKSPACE

bazel build -c opt --copt=-mavx tensorflow/examples/image_retraining:retrain

python /bazel-bin/tensorflow/examples/image_retraining/retrain.py \
   --image_dir /tmp/flower_photos  \
   --output_graph /path/output_graph.pb \
   --output_labels /path/output_labels.txt \
   --bottleneck_dir /path/bottleneck

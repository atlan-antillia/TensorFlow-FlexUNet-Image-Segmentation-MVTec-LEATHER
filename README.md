<h2>TensorFlow-FlexUNet-Image-Segmentation-MVTec-LEATHER (2025/10/22)</h2>
<!--
Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
 -->
This is the first experiment of Image Segmentation for <b>MVTec-LEATHER </b> based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512 pixels PNG dataset <a href="https://drive.google.com/file/d/1MgRkC38ty5y3kBxUtw7BSCr8AxbaMHk5/view?usp=sharing">
Augmented-MVTEC-LEATHER-ImageMask-Dataset.zip</a> with colorized masks 
 which was derived by us from <b>cable</b> subset of 
<a href="https://www.mvtec.com/company/research/datasets/mvtec-ad">
<b>The MVTec anomaly detection dataset
</a>
</b>
<br>
<br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of images and masks of <b>leather</b> subset of the MVTec dataset, 
we used our offline augmentation tool <a href="https://github.com/sarah-antillia/ImageMask-Dataset-Offline-Augmentation-Tool"> 
ImageMask-Dataset-Offline-Augmentation-Tool</a> to augment the <b>leather</b> subset.
<br><br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<b>rgb_map (color:blue, cut:red, fold:green,glue:yellow, poke:cyan)</b> <br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/images/color_3.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/masks/color_3.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test_output/color_3.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/images/deformed_alpha_1300_sigmoid_7_cut_17.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/masks/deformed_alpha_1300_sigmoid_7_cut_17.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test_output/deformed_alpha_1300_sigmoid_7_cut_17.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/images/deformed_alpha_1300_sigmoid_7_fold_6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/masks/deformed_alpha_1300_sigmoid_7_fold_6.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test_output/deformed_alpha_1300_sigmoid_7_fold_6.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1 Dataset Citation</h3>
The dataset used here was obtained from 
<br><br>
<a href="https://www.mvtec.com/company/research/datasets/mvtec-ad">
<b>The MVTec anomaly detection dataset
</a>
</b>
<br>
<h4>Citation</h4>
<b>If you use this dataset in scientific work, please cite our papers:</b><br>
Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: <br>
<b>The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;</b> <br>
in: International Journal of Computer Vision 129(4):1038-1059, 2021,<br>
 DOI: 10.1007/s11263-020-01400-4.<br>
<br>
Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: <br>
<b>MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; </b>in: <br>
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584-9592, 2019, <br>
DOI: 10.1109/CVPR.2019.00982.
<br>
<br>
<h4>LICENSE</h4>
The data is released under 
<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
</a>
<br>
<b>You may not use the material for commercial purposes.</b></a>
<br>
<br>
<h3>
2 MVTEC-LEATHER ImageMask Dataset
</h3>
<h3>2.1 ImageMask Dataset</h3>
 If you would like to train this MVTEC-LEATHER Segmentation model by yourself,
 please download our data <a href="https://drive.google.com/file/d/1MgRkC38ty5y3kBxUtw7BSCr8AxbaMHk5/view?usp=sharing">
 Augmented-MVTEC-LEATHER-ImageMask-Dataset.zip
 </a> on the google drive,
, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─MVTEC-LEATHER
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>MVTEC-LEATHER Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/MVTEC-LEATHER_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br><br>
<h3>2.2 ImageMask Dataset Generation</h3>
The folder structure of the original <b>mvtec_anomaly_detection</b> is the following.<br>
<pre>
./mvtec_anomaly_detection
├─bottle
├─cable
├─capsule
├─carpet
├─grid
├─hazelnut
├─leather
├─metal_nut
├─pill
├─screw
├─tile
├─toothbrush
├─transistor
├─wood
└─zipper
</pre>

For simplicity, we generated our Augmented MVTEC-LEATHER dataset from masks in <b>ground_truth</b> 
and images in <b>test</b> folders of <b>leather</b> dataset. 
<pre>
./leather
├─ground_truth
│  ├─color
│  ├─cut
│  ├─fold
│  ├─glue
│  └─poke
├─test
│  ├─color
│  ├─cut
│  ├─fold
│  ├─glue
│  ├─good
│  └─poke
└─train
    └─good
</pre>    
<br>
<br>
<h3>2.3 Train Images and Masks Sample </h3>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained MVTEC-LEATHER TensorFlowFlexUNet Model by using 
<a href="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/MVTEC-LEATHER, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 9

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 6
dropout_rate   = 0.03
dilation       = (3,3)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for MVTEC-LEATHER 1+5 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;                  color:blue, cut:red,      fold:green,  glue:yellow,    poke:cyan
rgb_map={(0,0,0):0,(0,0,255):1, (255,0,0):2, (0,255,0):3, (255,255, 0):4, (0,255,255):5, }

</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 23,24,25)</b><br>
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 47,48,49)</b><br>
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 49 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/asset/train_console_output_at_epoch49.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/MVTEC-LEATHER</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for MVTEC-LEATHER.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/asset/evaluate_console_output_at_epoch49.png" width="720" height="auto">
<br><br>Image-Segmentation-MVTEC-LEATHER

<a href="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this MVTEC-LEATHER/test was very low, and dice_coef_multiclass very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0083
dice_coef_multiclass,0.9959
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/MVTEC-LEATHER</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for MVTEC-LEATHER.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Images of 512x512 pixels </b><br>
<b>rgb_map (color:blue, cut:red, fold:green,glue:yellow, poke:cyan)</b> <br>

<table>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/images/deformed_alpha_1300_sigmoid_7_color_15.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/masks/deformed_alpha_1300_sigmoid_7_color_15.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test_output/deformed_alpha_1300_sigmoid_7_color_15.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/images/deformed_alpha_1300_sigmoid_8_cut_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/masks/deformed_alpha_1300_sigmoid_8_cut_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test_output/deformed_alpha_1300_sigmoid_8_cut_4.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/images/deformed_alpha_1300_sigmoid_8_fold_15.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/masks/deformed_alpha_1300_sigmoid_8_fold_15.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test_output/deformed_alpha_1300_sigmoid_8_fold_15.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/images/deformed_alpha_1300_sigmoid_8_glue_0.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/masks/deformed_alpha_1300_sigmoid_8_glue_0.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test_output/deformed_alpha_1300_sigmoid_8_glue_0.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/images/deformed_alpha_1300_sigmoid_9_poke_12.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/masks/deformed_alpha_1300_sigmoid_9_poke_12.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test_output/deformed_alpha_1300_sigmoid_9_poke_12.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/images/deformed_alpha_1300_sigmoid_8_fold_7.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test/masks/deformed_alpha_1300_sigmoid_8_fold_7.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MVTEC-LEATHER/mini_test_output/deformed_alpha_1300_sigmoid_8_fold_7.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. TensorFlow-FlexUNet-Image-Segmentation-Model</b>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>2. TensorFlow-FlexUNet-Image-Segmentation-MVTec-GRID</b>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-MVTec-GRID">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-MVTec-GRID
</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-MVTec-SCREW</b>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-MVTec-SCREW">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-MVTec-SCREW
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-MVTec-TRANSISTOR</b>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-MVTec-TRANSISTOR">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-MVTec-TRANSISTOR
</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-MVTec-CABLE</b>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-MVTec-CABLE">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-MVTec-CABLE
</a>


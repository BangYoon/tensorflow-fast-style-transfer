# Fast Style Transfer

A tensorflow implementation of fast style transfer described in the papers:
* [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/) by Johnson
* [Instance Normalization](https://arxiv.org/abs/1607.08022) by Ulyanov
  

## Usage

### Environment (2021.04.30)
Use GeForce RTX 2080 SUPER (CUDA version: 11.0)

* tensorflow         1.9.0 
* Pillow             8.2.0
* moviepy            1.0.2
* numpy              1.19.2
* scipy              1.1.0

### Train time

5~6 hours for training with 4 epochs and 6 batch size.


### Download
* Pretrained VGG19 file : [imagenet-vgg-verydeep-19.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) 
> Save the file under `pre_trained_model` 
* MSCOCO train2014 DB : [train2014.zip](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)  
> Extract images to `train2014`

### Train
```
python run_train.py --style <style file> --output <output directory> 
                    --trainDB <trainDB directory> --vgg_model <model directory>
```
*Example*:
`python run_train.py --style style/wave.jpg --output model --trainDB train2014 --vgg_model pre_trained_model`

#### Arguments
*Required* :  
* `--style`: Filename of the style image. *Default*: `images/wave.jpg`
* `--output`: File path for trained-model. Train-log is also saved here. *Default*: `models`
* `--trainDB`: Relative or absolute directory path to MSCOCO DB. *Default*: `train2014`
* `--vgg_model`: Relative or absolute directory path to pre trained model. *Default*: `pre_trained_model`

*Optional* :  
* `--content_weight`: Weight of content-loss. *Default*: `7.5e0`
* `--style_weight`: Weight of style-loss. *Default*: `5e2`
* `--tv_weight`: Weight of total-varaince-loss. *Default*: `2e2`
* `--content_layers`: *Space-separated* VGG-19 layer names used for content loss computation. *Default*: `relu4_2`
* `--style_layers`: *Space-separated* VGG-19 layer names used for style loss computation. *Default*: `relu1_1 relu2_1 relu3_1 relu4_1 relu5_1`
* `--content_layer_weights`: *Space-separated* weights of each content layer to the content loss. *Default*: `1.0`
* `--style_layer_weights`: *Space-separated* weights of each style layer to loss. *Default*: `0.2 0.2 0.2 0.2 0.2`
* `--max_size`: Maximum width or height of the input images. *Default*: `None`
* `--num_epochs`: The number of epochs to run. *Default*: `2`
* `--batch_size`: Batch size. *Default*: `4`
* `--learn_rate`: Learning rate for Adam optimizer. *Default*: `1e-3`
* `--checkpoint_every`: Save-frequency for checkpoint. *Default*: `1000`
* `--test`: Filename of the content image for *test during training*. *Default*: `None`
* `--max_size`: Maximum width or height of the input image for test. *None* do not change image size. *Default*: `None` 

#### Trained models
You can download all the 6 trained models from [here](https://mega.nz/#F!VEAm1CDD!ILTR1TA5zFJ_Cp9I5DRofg)

### Test  

```
python run_test.py --content <content file> --style_model <style-model file> --output <output file> 
```
*Example*:
`python run_test.py --content content/female_knight.jpg --style_model models/wave.ckpt --output result.jpg`

#### Arguments
*Required* :  
* `--content`: Filename of the content image. *Default*: `content/female_knight.jpg`
* `--style-model`: Filename of the style model. *Default*: `models/wave.ckpt`
* `--output`: Filename of the output image. *Default*: `result.jpg`  

*Optional* :  
* `--max_size`: Maximum width or height of the input images. *None* do not change image size. *Default*: `None`

## References
Tensorflow implementation : https://github.com/lengstrom/fast-style-transfer  



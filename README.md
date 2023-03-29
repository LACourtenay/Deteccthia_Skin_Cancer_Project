# Dettecthia Skin Cancer Project
Code used for the robust statistical analysis and development of deep learning tools for the study of Squamous Cell Carcinoma,
Basal Cell Carcinoma, and Actinic Keratosis patients, using Near Infrared Hyperspectral Imagery.

-----------------------------------------------------------------------------------------------------------------

## <b> Author Details </b>

<b> Author </b>: Lloyd A. Courtenay

<b> Email </b>: ladc1995@gmail.com

<b> ORCID </b>: https://orcid.org/0000-0002-4810-2001

<b> Current Afiliation </b>: Universidad de Salamanca [USAL]

---------------------------------------------------------------------------------------------------

This code has been designed for the open-source free R, Python and JavaScript programming languages.

---------------------------------------------------------------------------------------------------

## <b> Project Details </b>

All of the research included within the present repository was collected as part of the <b> DETTECTHIA </b> research project, funded by the
Junta de Castilla y Leon and the European Regional Development Fund, with project reference number: SA097P20.

<b> Project Title </b>:  Autmotaización en la <b>DETEC</b>ción <b>T</b>emprana y monitorización de <b>T</b>umores cutáneos
no-melanoma mediante imagen <b>H</b>iperspectral e <b>I</b>nteligencia <b>A</b>rtificial (<b>DETECTTHIA</b>)

<b> Primary Investigator </b>: Diego González-Aguilera

<b> Host Institution </b>: TIDOP Research Group - University of Salamanca.

<b> Website </b>: http://tidop.usal.es/

---------------------------------------------------------------------------------------------------

## <b> Repository Details </b>

The present repository contains:

* <b> Robust Statistics Folder </b>
  * Comma delimited table containing all the hyperspectral signatures obtained from cutaneous Squamous Cell Carcinoma (SCC) patients,
  Basal Cell Carcinoma (BCC) patients, and Actinic Keratosis (AK) patients. Photo ID's have been included so as to differentiate between
  signatures obtained from the same images, nevertheless, so as to ensure patient animosity, no further
  information has or will be provided.
  * A list of each of the Headwall Hyperspec NIR X Series' bands, and their frequencies (nm).
  * All R code for the analysis of this data, as well as R code for hyperspectral Kernel Principal Component Analysis.
  * <b> Results Folder </b>
    * Three .csv files containing the numeric results from the project.
    * A folder containing all JavaScript code used to create Amcharts figures for the visualisation of some data.
* <b> Deep Learning Folder </b>
  * Code for the training of CNSVM models.
  * Examples of pretrained models and weights
  * SSH file used to launch the algorithm in the Centre of Supercomputation of Castilla y León
  * <b> Transfer Learning Folder </b>
    * All the data that can be used for training of the original model as well as the transfer-learning trained model. Set up for the classification
    of H and BCC samples.
    * Python code used for Transfer Learning.
  * Source code for figures comparing algorithms and visualising example signatures

--------------------------------------------------------

## <b> System Requirements for Deep Learning </b>

* Python
    * Version 3.0 or higher
* Tensorflow
    * Version 2.0 or higher
* Numpy
* Scikit Learn
* Hyperopt
* Joblib

<i>If the user wishes to use GPU, then CUDA must also be installed and configured according to the requiremenets of Tensorflow</i>

--------------------------------------------------------

## <b> Citation </b>

Please cite this repository as:

 <b> Courtenay, L.A. (2023) Code and Data for the DETTECTHIA skin cancer project. https://github.com/LACourtenay/Dettecthia_Skin_Cancer_Project </b>

--------------------------------------------------------

Comments, questions, doubts, suggestions and corrections can all be directed to L. A. Courtenay at the email provided above.

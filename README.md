# Dettecthia Skin Cancer Project
Code used for the robust statistical analysis and development of deep learning tools for the study of Squamous Cell Carcinoma,
Basal Cell Carcinoma, and Actinic Keratosis patiens, using Near Infrared Hyperspectral Imagery.

-----------------------------------------------------------------------------------------------------------------

<i>
Author

Lloyd A. Courtenay

Email

ladc1995@gmail.com

ORCID

https://orcid.org/0000-0002-4810-2001

Current Afiliations:

Universidad de Salamanca [USAL]

</i>

---------------------------------------------------------------------------------------------------

This code has been designed for the open-source free R, Python and JavaScript programming languages.

---------------------------------------------------------------------------------------------------

## <b> Project Details </b>

All of the research included within the present repository was collected as part of the <b> DETTECTHIA </b> research project, funded by the
Junta de Castilla y Leon and the European Regional Development Fund, with project reference number: SA097P20.

<b> Project Title </b>:  Autmotaización en la <b>DETEC</b>ción Temprana y monitorización de Tumores cutáneos no-melanoma mediante imagen Hiperspectral e Inteligencia Artificia (DETECTTHIA)

TIDOP research group website: http://tidop.usal.es/

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

--------------------------------------------------------

<b>System Requirements for Deep Learning: </b>

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

Please cite this repository as:

 <b> Courtenay (2023) Code and Data for the Dettecthia skin cancer project. https://github.com/LACourtenay/Dettecthia_Skin_Cancer_Project </b>

--------------------------------------------------------

Comments, questions, doubts, suggestions and corrections can all be directed to L. A. Courtenay at the email provided above.

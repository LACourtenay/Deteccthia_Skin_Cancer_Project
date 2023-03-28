
# define the samples the user wishes to classify

sample1 = "H"
sample2 = "BCC"
file_name = "H_BCC"

from Dettecthia_CNSVM import CNSVM_Model
import pandas as pd

if __name__ == "__main__":
    
    df = pd.read_csv('DL_dataset.csv')

    df = df.loc[(df['Sample'] == sample1) | (df['Sample'] == sample2)]
    df = df.replace({sample1: 0, sample2: 1})
    df = df.to_numpy()
    
    model = CNSVM_Model()
    
    model.prepare_training_data(dataset = df)
    
    model.define_model(select_act = "relu", opt = "adam")
    
    model.fit_CNN_model(use_gpu = True)
    
    model.get_base_nn()
    
    model.tune_SVM()
    
    model.fit_SVM()
    
    model.evaluate_CNSVM()
    
    model.save_model(file_name)

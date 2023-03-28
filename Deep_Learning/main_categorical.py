
file_name = "categorical"

from Dettecthia_CNSVM import CNSVM_Model_categorical, mse_calc
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

if __name__ == "__main__":
    
    df = pd.read_csv('DL_dataset.csv')

    model = CNSVM_Model_categorical()
    
    model.prepare_training_data(dataset = df)
    
    model.define_model(select_act = "relu", opt = "adam")
    
    model.fit_CNN_model(use_gpu = True)
    
    model.get_base_nn()
    
    model.tune_SVM()
    
    model.fit_SVM()
    
    model.save_model("categorical_CNSVM_dettecthia")

    model.evaluate_CNSVM()
    
    model.save_model(file_name)
    
    predictions = model.NSVM.predict(model.X_neural_test)
    pred_probs = model.NSVM.predict_proba(model.X_neural_test)
    accuracy = classification_report(model.test_encoded_y, predictions)
    Kappa = cohen_kappa_score(model.test_encoded_y, predictions)
    confusion = confusion_matrix(model.test_encoded_y, predictions)
    print(accuracy)
    print(f"Kappa: {Kappa:.2f}")
    print(confusion)
    print(model.key)

    np.savetxt("categorical_probabilities.csv", pred_probs, delimiter=';')

    mse_calc(model.test_encoded_y,pred_probs)

    predicted_label = []
    for i in range(model.test_encoded_y.shape[0]):
        if model.test_encoded_y[i] == 0:
            predicted_label.append("BCC")
        elif model.test_encoded_y[i] == 1:
            predicted_label.append("H")
        elif model.test_encoded_y[i] == 2:
            predicted_label.append("AK")
        elif model.test_encoded_y[i] == 3:
            predicted_label.append("SCC")
 
    predicted_label

    dict={'name':predicted_label}
    predicted_labels = pd.DataFrame(dict) 
    predicted_labels.to_csv('predicted_labels.csv')


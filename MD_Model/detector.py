import numpy as np
from param_parser import parameter_parser
from joblib import load

def Predict(X,clf):
    '''
    param: X (feature vector)
    return: y (label), 0 for benign, 1 for malware
    '''
    model = load('./model_save/'+clf+'.joblib')
    label = model.predict(X)
    return label


def main(args):
    feature = np.load('../feature/feature.npy')
    result = Predict(feature,args.model)
    print(result)



if __name__=='__main__':
    args = parameter_parser()
    main(args)

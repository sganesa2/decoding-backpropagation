from data.dataset import NgramDataset
from model.train import BatchNormalizedMLP
from model.inference import run_batchnormlized_mlp

def main()->list[str]:
    dataset_obj = NgramDataset(3, "dataset.txt", 25626, 3204, 3203)
    _ , _ = dataset_obj.get_complete_dataset()
    x_train, y_train = dataset_obj.trainset

    model = BatchNormalizedMLP(200, 3, feature_dims=10)
    model.minibatch_gradient_descent(32, x_train, y_train, 1000, 0.1,0)

    words = run_batchnormlized_mlp(10, model, 'minibatch_gradient_descent','emm')
    return words

if __name__=="__main__":
   w = main()
   print(w)
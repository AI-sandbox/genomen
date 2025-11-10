import genomen.utils as utils
from genomen.data import DataSet, split, bootstrap
from genomen.model import GenomenModel

utils.set_config_path("config.yml")

dataset = DataSet()
train_set, test_set = split(dataset, test_size=0.2)

model = GenomenModel()
model.fit(train_set, test_set)

print("Saving model...")
model.save("model.pkl")

print("Loading model...")
model2 = GenomenModel.load("model.pkl")

print("Predicting...")
geno_preds, covar_preds, preds = model2.predict(test_set)


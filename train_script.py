import argparse
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# argumentos para o script
parser = argparse.ArgumentParser()
parser.add_argument('--max_depth', type=int, default=5)
parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--min_child_weight', type=float, default=1)
parser.add_argument('--subsample', type=float, default=0.8)
parser.add_argument('--gamma', type=float, default=0)
parser.add_argument('--num_round', type=int, default=100)
parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
args = parser.parse_args()

# define os parâmetros do modelo
param = {
    'max_depth': args.max_depth,
    'eta': args.eta,
    'min_child_weight': args.min_child_weight,
    'subsample': args.subsample,
    'gamma': args.gamma,
    'objective': 'reg:squarederror',
    'verbosity': 0
}

# converte os dados para o formato XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
dval = xgb.DMatrix(X_validation, label=y_validation)

# treinamento do modelo
num_round = args.num_round
bst = xgb.train(param, dtrain, num_round, evals=[(dval, 'validation')])

# realiza as previsões
preds = bst.predict(dtest)

# avaliação do modelo
rmse = mean_squared_error(y_test, preds, squared=False)

# salva o modelo
model_dir = '/opt/ml/model'
os.makedirs(model_dir, exist_ok=True)
bst.save_model(os.path.join(model_dir, 'xgboost-model'))
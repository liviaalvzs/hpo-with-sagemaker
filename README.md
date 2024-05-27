# Hyperparameter Optimization (HPO) - XGBoost e SageMaker 🔍🚀

Artigo no Medium: [Explorando o Potencial do Amazon SageMaker na Otimização de Parâmetros para um Modelo XGBoost](https://medium.com/@alvzslivia/explorando-o-potencial-do-amazon-sagemaker-na-otimiza%C3%A7%C3%A3o-de-par%C3%A2metros-para-um-modelo-xgboost-b8ed136efee7).

Este projeto explora o potencial do Amazon SageMaker na otimização de parâmetros para um modelo XGBoost. O objetivo principal é maximizar o desempenho do modelo ajustando os hiperparâmetros de forma automatizada. 

Inicialmente, gera-se um conjunto de dados sintéticos que simulam características imobiliárias.Em seguida, normaliza-se os dados e os divide-se em conjuntos de treinamento, teste e validação. Essa etapa é importante pois permite avaliar o desempenho do modelo de forma precisa.

Utiliza-se o Amazon SageMaker para configurar e treinar um modelo XGBoost. 

Para otimizar os hiperparâmetros do modelo XGBoost, emprega-se o SageMaker Hyperparameter Tuner. Este componente automatiza o processo de busca pelos melhores hiperparâmetros, permitindo que o modelo alcance seu máximo potencial de desempenho (dentro dos intervalos inseridos).

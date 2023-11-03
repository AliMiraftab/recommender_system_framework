import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import lightgbm as lgb
from mlflow.models.signature import infer_signature
import time

training_features = []

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict(model_input)

def train_model(pdf_features):

    run_name = 'whatnot_recsys'
    model_name = 'lambdarank'

    target_label = ['label'] # label = purchase + offer + click

    pdf_train = pdf_features[pdf_features.train_data]
    pdf_validation = pdf_features[pdf_features.validation_data] 
    pdf_test = pdf_features[pdf_features.test_data] 

    qids_train = pdf_train.groupby("request_response_id")["request_response_id"].count().to_numpy()
    X_train = pdf_train[training_features]
    y_train = pdf_train[target_label]

    qids_validation = pdf_validation.groupby("request_response_id")["request_response_id"].count().to_numpy()
    X_validation = pdf_validation[training_features]
    y_validation = pdf_validation[target_label]

    qids_test = pdf_test.groupby("request_response_id")["request_response_id"].count().to_numpy()
    X_test = pdf_test[training_features]
    y_test = pdf_test[target_label]    


    with mlflow.start_run(run_name=run_name):   
        lambdarank = lgb.LGBMRanker(objective="lambdarank",
                                    metric="ndcg",
                                    n_estimators=500,
                                    learning_rate=0.05, 
                                    label_gain=[0, 1, 3, 7],
                                    lambdarank_position_bias_regularization=0.2)
        lambdarank.fit(X=X_train,\
                        y=y_train,
                        group=qids_train,
                        eval_set=[(X_validation, y_validation), (X_train, y_train)],
                        eval_group=[qids_validation, qids_train],
                        eval_at=[1, 5, 20, 50, 100, 500],
                        verbose=10)
        
        mlflow.log_metric('ndcg_at1', lambdarank.best_score_['valid_0']['ndcg@1'])
        mlflow.log_metric('ndcg_at5', lambdarank.best_score_['valid_0']['ndcg@5'])
        mlflow.log_metric('ndcg_at20', lambdarank.best_score_['valid_0']['ndcg@20'])
        mlflow.log_metric('ndcg_at50', lambdarank.best_score_['valid_0']['ndcg@50'])
        mlflow.log_metric('ndcg_at100', lambdarank.best_score_['valid_0']['ndcg@100'])
        mlflow.log_metric('ndcg', lambdarank.best_score_['valid_0']['ndcg@500'])
        

        lambdarank_params = lambdarank.get_params(deep=True)
        for key in lambdarank_params.keys():
            mlflow.log_param(key, lambdarank_params[key])
        wrappedModel = SklearnModelWrapper(lambdarank)
        signature = infer_signature(X_train, wrappedModel.predict(None, X_validation))
        mlflow.pyfunc.log_model("lambdarank", python_model=wrappedModel, signature=signature)
        
        run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "best_bets_rt_mvp"').iloc[0].run_id    
        model_version = mlflow.register_model(f"runs:/{run_id}/lambdarank", model_name)
        time.sleep(15)    
    return 

# def hyper_parameter_tuning_lambdarank():
#     pdf_training_and_validation = pdf_features[~pdf_features.test_data]
#     pdf_testing = pdf_features[pdf_features.test_data] 

#     pdf_training = pdf_training_and_validation[~pdf_training_and_validation.val_data]
#     x_train = pdf_training[model_feats]
#     y_train = (pdf_training['booked'] + pdf_training['offered'] + pdf_training['clicked'])
#     qids_train = pdf_training.groupby("page_id", sort=False)["page_id"].count().to_numpy()

#     pdf_validation = pdf_training_and_validation[pdf_training_and_validation.val_data]
#     x_validation = pdf_validation[model_feats]
#     y_validation = (pdf_training['booked'] + pdf_training['offered'] + pdf_training['clicked'])
#     qids_validation = pdf_validation.groupby("page_id", sort=False)["page_id"].count().to_numpy()

#     x_test = pdf_testing[model_feats]
#     y_test = (pdf_training['booked'] + pdf_training['offered'] + pdf_training['clicked'])

#     save_root="/dbfs/tmp/ali/bfs_optimization/"
#     save_root_s3="s3://apiary-analytics-927134741764-us-east-1-my-datascience/ali/projects/bfs_optimization/models/"
#     # num_tries=2

#     ps=iter(ParameterSampler({'objective':['lambdarank'],
#                             'metric':['ndcg'],
#                             'learning_rate':[10**tmp for tmp in pl.linspace(np.log10(learning_rate_range[0]),np.log10(learning_rate_range[1]),100)],
#                             'boosting_type':boosting_type_values,
#                             'num_leaves': list(range(num_leaves_range[0],num_leaves_range[1])),
#                             'bagging_fraction':pl.linspace(bagging_fraction_range[0],bagging_fraction_range[1],50),
#                             'bagging_freq':bagging_freq_values,
#                             'feature_fraction':feature_fraction_values,
#                             'n_estimators': list(range(n_estimators_range[0],n_estimators_range[1])),
#                             'max_depth':[-1] + list(range(max_depth_range[0],max_depth_range[1])),
#                             'label_gain':[label_gain]},n_iter=num_tries))

#     model_description = 'lambdarank_wo_price'
#     iteration = itertools.count(1)
#     for _ in ps:
#     iteration_next = next(iteration)
#     print(f'Running teration: {iteration_next}')
#     parameters = json.dumps(next(ps))
#     with mlflow.start_run(run_name=f"{model_description}_{iteration_next}_{get_month_day_hour_minute_of_now_utc()}"):
#         trained_model = (lgb.LGBMRanker(**json.loads(parameters))
#                         .fit(X=x_train.drop(columns='price_displayed_pmindelta'),
#                             y=y_train,
#                             group=qids_train,
#                             eval_set=[(x_validation.drop(columns='price_displayed_pmindelta'), y_validation), 
#                                     (x_train.drop(columns='price_displayed_pmindelta'), y_train)],
#                             eval_group=[qids_validation, qids_train],
#                             eval_at=10,
#                             verbose=10))
#         model_name_to_save = f'{model_description}_{iteration_next}_{get_month_day_hour_minute_of_now_utc()}.pkl'
#         try:
#         pickle.dump(trained_model,open(f"{save_root}{trained_model}","wb+"))
#         except FileNotFoundError:
#         os.mkdir(save_root)
#         pickle.dump(trained_model,open(f"{save_root}{trained_model}","wb+"))

#         # Validation score:
#         val_avg_booked_rank=(
#         spark.createDataFrame(pdf_validation.assign(score=trained_model.predict(pdf_validation[trained_model.feature_name_])))
#         .withColumn("rank",F.row_number().over(Window.partitionBy("page_id").orderBy(F.desc("score"))))
#         .where("booked=1")
#         .selectExpr("avg(rank) as avgrank").collect()[0].avgrank
#         )

#         # Logging
#         mlflow.log_metric("val_avg_booked_rank", val_avg_booked_rank)
#         mlflow.log_params(json.loads(parameters))
#         mlflow.set_tag("model_path",f"{save_root}{trained_model}")
#         mlflow.set_tag("model_features",str(trained_model.feature_name_))
#     return 
from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics

@dsl.container_component
def data_ingestion(
    param_file_path: str,
    data_url: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    )-> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image='prakash3112/kubeflow-pipeline:ingestv3',
        command=['python', '/app/ingest.py'],
        args=[
            param_file_path,
            data_url,
            train_data.path,
            test_data.path
        ]
    )

@dsl.container_component
def data_preprocessing(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    text_column: str,
    target_column: str,
    train_processed: Output[Dataset],
    test_processed: Output[Dataset],
)-> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image='prakash3112/kubeflow-pipeline:preprocess-v10',
        command=['python', '/app/preprocess.py'],
        args=[
            train_data.path,
            test_data.path,
            train_processed.path,
            test_processed.path,
            text_column,
            target_column
        ]
    )



@dsl.container_component
def feature_engineering(
    param_file_path: str,
    train_processed: Input[Dataset],
    test_processed: Input[Dataset],
    train_tfidf: Output[Dataset],
    test_tfidf: Output[Dataset],
)-> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image='prakash3112/kubeflow-pipeline:feature_engineering-v6',
        command=['python', '/app/feature_engineering.py'],
        args=[
            param_file_path,
            train_processed.path,
            test_processed.path,
            train_tfidf.path,
            test_tfidf.path
        ]
    )

@dsl.container_component
def train_model(
    param_file_path: str,
    train_tfidf: Input[Dataset],
    model: Output[Model],
)-> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image='prakash3112/kubeflow-pipeline:train-v4',
        command=['python', '/app/model_training.py'],
        args=[param_file_path, 
              train_tfidf.path, 
              model.path],
    )


@dsl.container_component
def evaluate_model(
    model: Input[Model],
    test_tfidf: Input[Dataset],
    metrics: Output[Metrics],
)-> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image='prakash3112/kubeflow-pipeline:model_evaluation-v9',
        command=['python', '/app/model_evaluation.py'],
        args=[
            model.path,           # ✔ model_load_path
            test_tfidf.path,      # ✔ test_data_path
            metrics.path
        ]
    )

@dsl.pipeline(name='spam-detection-pipeline', 
              description='Pipeline for spam detection using TF-IDF and RandomForest',
              pipeline_root='minio://mlpipeline/artifacts'
              )
def spam_detection_pipeline(
    param_file_path: str = '/app/params.yaml',
    data_url: str = 'https://raw.githubusercontent.com/PrakashD2003/DATASETS/main/spam.csv',
    text_column: str = 'text',
    target_column: str = 'target'
    ):

    ingest_op = data_ingestion(param_file_path=param_file_path,
                               data_url=data_url)
    
    preprocess_op = data_preprocessing(
        train_data=ingest_op.outputs['train_data'],
        test_data=ingest_op.outputs['test_data'],
        text_column=text_column,
        target_column=target_column
    )

    feature_op = feature_engineering(
        param_file_path=param_file_path,
        train_processed=preprocess_op.outputs['train_processed'],
        test_processed=preprocess_op.outputs['test_processed']
    )

    train_op = train_model(
        param_file_path=param_file_path,
        train_tfidf=feature_op.outputs['train_tfidf']
    )

    evaluate_model(
        model=train_op.outputs['model'],
        test_tfidf=feature_op.outputs['test_tfidf']
    )

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=spam_detection_pipeline,
        package_path='spam_detection_pipeline.yaml'
    )
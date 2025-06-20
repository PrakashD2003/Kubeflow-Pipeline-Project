from kfp import dsl, compiler

@dsl.container_component
def data_ingestion(param_file_path:str, data_url:str)
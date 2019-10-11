import subprocess
import sagemaker
from sagemaker.pytorch import PyTorch

instance_type = 'ml.p2.xlarge'
train_data_path = 's3://prj-kyocera/research'
output_path = 's3://prj-kyocera/research/output'
code_location = 's3://prj-kyocera/research/src'
role = "arn:aws:iam::533155507761:role/service-role/AmazonSageMaker-ExecutionRole-20190312T160681"
source_dir = "."
hyperparams = {'configs': "./configs/graphkv_sm.json", 'preprocess': True}

tf_estimator = PyTorch(entry_point='./process/train.py',
                            source_dir=".",
                            code_location=code_location,
                            output_path=output_path,
                            role=role, #sagemaker.get_execution_role(),
                            train_instance_type=instance_type,
                            train_instance_count=1,
                            base_job_name='hector-kyocera-chargrid',
                            train_max_run=12*60*60,
                            train_volume_size=20,
                            framework_version='1.12',
                            train_use_spot_instances=True,
                            train_max_wait=14*60*60,
                            py_version="py3",
                            hyperparameters=hyperparams,
                            script_mode=True
                        )
tf_estimator.fit({'train': train_data_path})
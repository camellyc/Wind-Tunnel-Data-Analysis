python ./preprocess_braidz_and_optotrigger.py $DATA preprocessing_parameters.yaml
python ./trim_braidz_and_optotrigger.py $DATA preprocessing_parameters.yaml
python ./generate_diagnostic_plots.py $DATA
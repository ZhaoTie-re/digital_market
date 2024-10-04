params.input_dir = '/Users/tie_zhao/Desktop/digital_market/DataSrc'
params.result_dir = '/Users/tie_zhao/Desktop/digital_market/Results'
params.scriptPath = '/Users/tie_zhao/Desktop/digital_market/Scripts'
params.feature_ignore = '/Users/tie_zhao/Desktop/digital_market/DataSrc/ignore_feature.txt'
params.targetCol = 'Conversion'

Channel
    .fromPath("${params.input_dir}/digital_marketing_campaign_dataset.csv")
    .map { item -> ['raw_data_check', item]}
    .set { raw_data_check_ch }

process raw_data_check {

    tag "${step_name}"
    conda "/Users/tie_zhao/miniconda3/envs/digital_market"

    publishDir "${params.result_dir}/01.raw_data_check", mode: 'symlink'

    input:
    tuple val(step_name), path(csv) from raw_data_check_ch
    val(target) from params.targetCol

    output:
    tuple val('feature_engineering'), file(h5) into feature_engineering_ch
    file '*.pdf'

    script:
    h5 = "raw_data_check.h5"
    """
    python ${params.scriptPath}/raw_data_check.py --csvPath ${csv} --targetCol ${target}
    """
}

process feature_engineering {

    tag "${step_name}"
    conda "/Users/tie_zhao/miniconda3/envs/digital_market"

    publishDir "${params.result_dir}/02.feature_engineering", mode: 'symlink'

    input:
    tuple val(step_name), path(inpu1_h5) from feature_engineering_ch
    path ignore_feature from params.feature_ignore

    output:
    tuple val('model_training'), file(output_h5) into model_training_ch1, model_training_ch2
    file '*.pdf'

    script:
    output_h5 = "feature_engineering.h5"
    """
    python ${params.scriptPath}/feature_engineer.py --h5Path ${inpu1_h5} --ignoreFeature ${ignore_feature}
    """
}

model_training_ch1
    .map { item -> [item[0], 'nn_mlp', item[1]] }
    .set {model_training_nn_mlp_ch}

process model_training_nn_mlp {

    tag "${step_name}:${model_name}"
    conda "/Users/tie_zhao/miniconda3/envs/digital_market"

    publishDir "${params.result_dir}/03.model_training/${model_name}", mode: 'symlink'

    input:
    tuple val(step_name), val(model_name), path(input_h5) from model_training_nn_mlp_ch

    output:
    tuple val('model_validation'), val('nn_mlp'), file(h5_model) into model_test_nn_mlp_ch
    file(best_result_json) into summary_nn_mlp_ch
    file(keras_model)
    file '*.pdf'

    script:
    keras_model = 'best_' + model_name + '.keras'
    h5_model = 'best_' + model_name + '.h5'
    best_result_json = 'best_results_' + model_name + '.json'
    """
    cp ${input_h5} copied_input.h5
    python ${params.scriptPath}/establish_${model_name}.py --h5Path copied_input.h5 --threshold 0.5
    """
}

model_training_ch2
    .map { item -> [item[0], 'xgb_classifier', item[1]] }
    .set {model_training_xgb_classifier_ch}

process model_training_xgb_classifier { 

    tag "${step_name}:${model_name}"
    conda "/Users/tie_zhao/miniconda3/envs/digital_market"

    publishDir "${params.result_dir}/03.model_training/${model_name}", mode: 'symlink'

    input:
    tuple val(step_name), val(model_name), path(input_h5) from model_training_xgb_classifier_ch

    output:
    tuple val('model_validation'), val('xgb_classifier'), file(pkl_model) into model_test_xgb_classifier_ch
    file(best_result_json) into summary_xgb_classifier_ch
    file(joblib_model)
    file '*.pdf'

    script:
    joblib_model = 'best_' + model_name + '.joblib'
    pkl_model = 'best_' + model_name + '.pkl'
    best_result_json = 'best_results_' + model_name + '.json'
    """
    cp ${input_h5} copied_input.h5
    python ${params.scriptPath}/establish_${model_name}.py --h5Path copied_input.h5 --threshold 0.7
    """
}


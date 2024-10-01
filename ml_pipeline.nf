params.input_dir = '/Users/tie_zhao/Desktop/digital_market/DataSrc'
params.result_dir = '/Users/tie_zhao/Desktop/digital_market/Results'
params.scriptPath = '/Users/tie_zhao/Desktop/digital_market/Scripts'
params.feature_ignore = '/Users/tie_zhao/Desktop/digital_market/DataSrc/ignore_feature.txt'

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

    output:
    tuple val('feature_engineering'), file(h5) into feature_engineering_ch
    file '*.pdf'

    script:
    h5 = "raw_data_check.h5"
    """
    python ${params.scriptPath}/raw_data_check.py --csvPath ${csv}
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
    tuple val('model_training'), file(output_h5) into model_training_ch
    file '*.pdf'

    script:
    output_h5 = "feature_engineering.h5"
    """
    python ${params.scriptPath}/feature_engineer.py --h5Path ${inpu1_h5} --ignoreFeature ${ignore_feature}
    """
}
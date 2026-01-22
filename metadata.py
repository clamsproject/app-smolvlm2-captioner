"""
The purpose of this file is to define the metadata of the app with minimal imports. 

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # first set up some basic information
    metadata = AppMetadata(
        name="SmolVLM2 Captioner",
        description="Applies SmolVLM2-2.2B-Instruct multimodal model to video frames for image captioning.",
        app_license="Apache 2.0",
        identifier="smolvlm2-captioner",
        url="https://github.com/clamsproject/app-smolvlm2-captioner"
    )

    # and then add I/O specifications: an app must have at least one input and one output
    metadata.add_input(DocumentTypes.VideoDocument)
    metadata.add_input(DocumentTypes.ImageDocument)
    metadata.add_input(AnnotationTypes.TimeFrame)
    metadata.add_output(AnnotationTypes.Alignment)
    metadata.add_output(DocumentTypes.TextDocument)
    
    # (optional) and finally add runtime parameter specifications
    metadata.add_parameter(
        name='frameInterval', type='integer', default=30,
        description='The interval at which to extract frames from the video if there are no timeframe annotations. '
        'Default is every 30 frames.'
    )
    metadata.add_parameter(
        name='defaultPrompt', type='string', default='Describe what is shown in this video frame. Analyze the purpose of this frame in the context of a news video. Transcribe any text present.',
        description='default prompt to use for timeframes not specified in the promptMap. If set to `-`, '
                     'timeframes not specified in the promptMap will be skipped.'
    )
    metadata.add_parameter(
        name='promptMap', type='map', default=[],
        description=('mapping of labels of the input timeframe annotations to new prompts. Must be formatted as '
                     '\"IN_LABEL:PROMPT\" (with a colon). To pass multiple mappings, use this parameter multiple '
                     'times. By default, any timeframe labels not mapped to a prompt will be used with the default'
                     'prompt. In order to skip timeframes with a particular label, pass `-` as the prompt value.'
                     'in order to skip all timeframes not specified in the promptMap, set the defaultPrompt'
                     'parameter to `-`'))
    
    metadata.add_parameter(
        name='defaultSystemPrompt', type='string', default='',
        description='default system prompt to use for all timeframes. System prompts are passed to the model using the '
                   'messages format with role="system", providing context or instructions that guide the model\'s behavior. '
                   'The processor will format this properly using its chat template.'
    )
    metadata.add_parameter(
        name='systemPromptMap', type='map', default=[],
        description=('mapping of labels of the input timeframe annotations to system prompts. Must be formatted as '
                     '\"IN_LABEL:SYSTEM_PROMPT\" (with a colon). To pass multiple mappings, use this parameter multiple '
                     'times. System prompts are passed to the model using the messages format with role="system", '
                     'providing context or instructions that guide the model\'s behavior.'))
    

    # add parameter for config file name
    metadata.add_parameter(
        name='config', type='string', default="config/default.yaml", description='Name of the config file to use.'
    )
    
    # add parameter for num_beams
    metadata.add_parameter(
        name='num_beams', type='integer', default=1,
        description='Number of beams for beam search during text generation. Default is 1. '
                    'Higher values may improve quality but increase generation time.'
    )
    
    # add parameter for batch_size
    metadata.add_parameter(
        name='batchSize', type='integer', default=12,
        description='Number of images to process in each batch. Default is 12. '
                    'Higher values may improve throughput but require more memory.'
    )

    # add parameter for allRepresentatives
    metadata.add_parameter(
        name='allRepresentatives', type='boolean', default=False,
        description='When true, process all representative TimePoints in each TimeFrame instead of just the first one. '
                    'This allows captioning multiple frames per TimeFrame when the TimeFrame has multiple representatives. '
                    'Default is false (only the first representative is processed).'
    )

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))

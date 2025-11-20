import argparse
import logging
import yaml
import torch
from pathlib import Path
import tqdm
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from clams import ClamsApp, Restifier
from clams.appmetadata import AppMetadata
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh


class SmolVLM2Captioner(ClamsApp):

    def __init__(self):
        super().__init__()
        # Check if CUDA is available
        if torch.cuda.is_available():
            self.device = "cuda"
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            self.logger.info("Using CPU")
        
        model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        self.logger.info(f"Loading model from {model_path}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # --- CRITICAL CONFIGURATION ---
        # Decoder-only models must use left-padding for generation
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        # ------------------------------

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        )
        
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        self.logger.info("SmolVLM2 model loaded successfully")

    def _appmetadata(self) -> AppMetadata:
        pass
    
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_prompt(self, label: str, parameters: dict) -> str:
        if 'promptMap' in parameters and parameters['promptMap']:
            for mapping in parameters['promptMap']:
                if ':' in mapping:
                    map_label, map_prompt = mapping.split(':', 1)
                    if map_label == label:
                        return map_prompt
        if 'defaultPrompt' in parameters:
            return parameters['defaultPrompt']
        return ""

    def get_system_prompt(self, label: str, parameters: dict) -> str:
        if 'systemPromptMap' in parameters and parameters['systemPromptMap']:
            for mapping in parameters['systemPromptMap']:
                if ':' in mapping:
                    map_label, map_prompt = mapping.split(':', 1)
                    if map_label == label:
                        return map_prompt
        if 'defaultSystemPrompt' in parameters:
            return parameters['defaultSystemPrompt']
        return ""

    def get_prompts(self, label: str, parameters: dict):
        """Get system and user prompts separately for a given label."""
        system_prompt = self.get_system_prompt(label, parameters)
        user_prompt = self.get_prompt(label, parameters)
        return (system_prompt or "", user_prompt or "")

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug(f"Annotating with parameters: {parameters}")
        config_file = parameters.get('config')
        
        if config_file:
            config_dir = Path(__file__).parent
            config_file_path = config_dir / config_file
            config = self.load_config(config_file_path)
            if 'default_prompt' in config:
                parameters['defaultPrompt'] = config['default_prompt']
            if 'custom_prompts' in config:
                prompt_map = []
                for label, prompt in config['custom_prompts'].items():
                    prompt_map.append(f"{label}:{prompt}")
                parameters['promptMap'] = prompt_map
            if 'default_system_prompt' in config:
                parameters['defaultSystemPrompt'] = config['default_system_prompt']
            if 'custom_system_prompts' in config:
                system_prompt_map = []
                for label, prompt in config['custom_system_prompts'].items():
                    system_prompt_map.append(f"{label}:{prompt}")
                parameters['systemPromptMap'] = system_prompt_map
        else:
            config = {}
        
        if 'context_config' not in config:
            config['context_config'] = {
                'input_context': 'timeframe',
                'timeframe': {
                    'app_uri': 'http://apps.clams.ai/swt-detection/',
                    'label_mapping': {},
                    'ignore_other_labels': False
                }
            }
            
        batch_size = parameters.get('batchSize', 12)

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)

        def process_batch(prompts_batch, images_batch, annotations_batch):
            """
            Processes a batch of images simultaneously using the chat template.
            """
            try:
                conversations_batch = []

                # 1. Build the conversation structure for every item in the batch
                for (system_prompt, user_prompt), image in zip(prompts_batch, images_batch):
                    messages = []
                    
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": [{"type": "text", "text": system_prompt}]
                        })
                    
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": user_prompt}
                        ]
                    })
                    conversations_batch.append(messages)

                # 2. Apply chat template
                inputs = self.processor.apply_chat_template(
                    conversations_batch,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    padding=True,
                    return_tensors="pt"
                )

                # 3. Move to device
                inputs = inputs.to(self.device)
                
                # 4. Generate
                generated_ids = self.model.generate(**inputs, max_new_tokens=200)
                
                # 5. STRIP PROMPT FROM OUTPUT
                # Calculate length of input tokens to slice the generated_ids
                input_len = inputs.input_ids.shape[1]
                # Slice to get only new tokens
                generated_ids_trimmed = generated_ids[:, input_len:]
                
                # 6. Decode only the new tokens
                generated_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

                # 7. Create Annotations
                for i, text in enumerate(generated_texts):
                    annotation = annotations_batch[i]
                    clean_text = text.strip()
                    
                    text_document = new_view.new_textdocument(
                        text=clean_text,
                        document=annotation.get('document_id'),
                        origin=annotation.get('origin_id'),
                        provenance='derived',
                        mime='application/json'
                    )
                    alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                    alignment.add_property("source", annotation['source'])
                    alignment.add_property("target", text_document.long_id)
                    
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        input_context = config['context_config']['input_context']

        # --- IMAGE DOCUMENT MODE ---
        if input_context == "image":
            image_docs = mmif.get_documents_by_type(DocumentTypes.ImageDocument)
            for i in range(0, len(image_docs), batch_size):
                batch_docs = image_docs[i:i + batch_size]
                prompts = [self.get_prompts('default', parameters)] * len(batch_docs)
                images = [Image.open(doc.location_path()) for doc in batch_docs]
                annotations_batch = [{'source': doc.long_id, 'document_id': doc.id, 'origin_id': doc.long_id} for doc in batch_docs]
                
                process_batch(prompts, images, annotations_batch) 

        # --- TIMEFRAME MODE ---
        elif input_context == 'timeframe':
            app_uri = config['context_config']['timeframe']['app_uri']
            all_views = mmif.get_all_views_contain(AnnotationTypes.TimeFrame)
            timeframes = []
            
            for view in all_views:
                if app_uri in view.metadata.app:
                    timeframes = list(view.get_annotations(AnnotationTypes.TimeFrame))
                    break
                    
            label_mapping = config['context_config']['timeframe'].get('label_mapping', {})
            ignore_other_labels = config['context_config']['timeframe'].get('ignore_other_labels', False)

            if ignore_other_labels:
                timeframes = [tf for tf in timeframes if tf.get_property('label') in label_mapping]
                if not timeframes:
                    self.logger.warning("No timeframes found matching label_mapping")
                    return mmif

            for timeframe in timeframes:
                timeframe.add_property('timeUnit', 'milliseconds')
            
            all_frame_numbers = [vdh.get_representative_framenum(mmif, timeframe) for timeframe in timeframes]
            video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]

            # Extract all images first
            all_images = vdh.extract_frames_as_images(video_doc, all_frame_numbers, as_PIL=True)

            for batch_idx in tqdm.tqdm(range(0, len(timeframes), batch_size)):
                batch_timeframes = timeframes[batch_idx:batch_idx + batch_size]
                batch_images = all_images[batch_idx:batch_idx + batch_size]
                
                prompts = []
                annotations_batch = []
                
                for idx_in_batch, timeframe in enumerate(batch_timeframes):
                    label = timeframe.get_property('label')
                    mapped_label = label_mapping.get(label, 'default')
                    prompts.append(self.get_prompts(mapped_label, parameters))
                    
                    # FIX: Do NOT create a new TimePoint. 
                    # Use the existing TimeFrame ID from the previous view as source.
                    annotations_batch.append({
                        'source': timeframe.long_id,
                        'document_id': video_doc.id,
                        'origin_id': timeframe.long_id
                    })

                process_batch(prompts, batch_images, annotations_batch)

        # --- FIXED WINDOW MODE ---
        elif input_context == 'fixed_window':
            video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
            stride = config['context_config']['fixed_window']['stride']
            
            try:
                fps = float(video_doc.get_property('fps'))
            except:
                fps = 29.97
            try:
                total_frames = int(video_doc.get_property('frameCount'))
            except:
                total_frames = int(29.97*60*60)
                
            frame_numbers = list(range(0, total_frames, int(fps * stride)))
            
            prompts = []
            images_batch = []
            annotations_batch = []
            
            for frame_number in tqdm.tqdm(frame_numbers):
                try:
                    image = vdh.extract_frames_as_images(video_doc, [frame_number], as_PIL=True)[0]
                except:
                    continue
                    
                prompts.append(self.get_prompts('default', parameters))
                images_batch.append(image)
                
                # For fixed window, we DO create TimePoints because they don't exist in input
                timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
                timepoint.add_property("timePoint", frame_number)
                annotations_batch.append({
                    'source': timepoint.long_id,
                    'document_id': video_doc.id,
                    'origin_id': timepoint.long_id
                })
                
                if len(prompts) == batch_size:
                    process_batch(prompts, images_batch, annotations_batch)
                    prompts, images_batch, annotations_batch = [], [], []

        else:
            raise ValueError(f"Unsupported input context: {input_context}")
            
        return mmif

def get_app():
    return SmolVLM2Captioner()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()
    app = SmolVLM2Captioner()
    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
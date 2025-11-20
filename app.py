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
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16)
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
        """Get system and user prompts separately for a given label.
        
        Returns:
            tuple: (system_prompt, user_prompt) where either can be empty string
        """
        system_prompt = self.get_system_prompt(label, parameters)
        user_prompt = self.get_prompt(label, parameters)
        return (system_prompt or "", user_prompt or "")

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug(f"Annotating with parameters: {parameters}")
        config_file = parameters.get('config')
        self.logger.debug(f"config_file: {config_file}")
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
        batch_size = 32
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)

        def process_batch(prompts_batch, images_batch, annotations_batch):
            # PyTorch model inference
            for (system_prompt, user_prompt), image, annotation in zip(prompts_batch, images_batch, annotations_batch):
                    try:
                        # Format prompt with system prompt - SmolVLM2 processor expects <image> token in text
                        # We'll combine system and user prompts but keep <image> token for processor
                        if system_prompt:
                            formatted_prompt = f"{system_prompt}\n\n<image> {user_prompt}"
                        else:
                            formatted_prompt = f"<image> {user_prompt}"
                        
                        inputs = self.processor(images=image, text=formatted_prompt, return_tensors="pt")
                        # Move inputs to the correct device
                        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        generated_ids = self.model.generate(**inputs, max_new_tokens=200)
                        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                        text_document = new_view.new_textdocument(
                            text=generated_text,
                            document=annotation.get('document_id'),
                            origin=annotation.get('origin_id'),
                            provenance='derived',
                            mime='application/json'
                        )
                        alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                        alignment.add_property("source", annotation['source'])
                        alignment.add_property("target", text_document.long_id)
                    except Exception as e:
                        self.logger.error(f"Error processing image: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        continue

        input_context = config['context_config']['input_context']

        if input_context == "image":
            image_docs = mmif.get_documents_by_type(DocumentTypes.ImageDocument)
            for i in range(0, len(image_docs), batch_size):
                batch_docs = image_docs[i:i + batch_size]
                prompts = [self.get_prompts('default', parameters)] * len(batch_docs)
                images = [Image.open(doc.location_path()) for doc in batch_docs]
                annotations_batch = [{'source': doc.long_id, 'document_id': doc.id, 'origin_id': doc.long_id} for doc in batch_docs]
                start_time = time.time()
                process_batch(prompts, images, annotations_batch)
                self.logger.debug(f"Processed batch of {len(batch_docs)} in {time.time() - start_time:.2f} seconds")

        elif input_context == 'timeframe':
            print(f"DEBUG: input_context: {input_context}", flush=True)
            self.logger.debug(f"input_context: {input_context}")
            app_uri = config['context_config']['timeframe']['app_uri']
            print(f"DEBUG: Looking for app_uri: {app_uri}", flush=True)
            all_views = mmif.get_all_views_contain(AnnotationTypes.TimeFrame)
            print(f"DEBUG: Found {len(all_views)} views with TimeFrame", flush=True)
            for view in all_views:
                self.logger.debug(f"view.metadata.app: {view.metadata.app}")
                if app_uri in view.metadata.app:
                    self.logger.debug(f"found view with app_uri: {app_uri}")
                    timeframes = view.get_annotations(AnnotationTypes.TimeFrame)
                    timeframes_list = list(timeframes)
                    print(f"DEBUG: Found {len(timeframes_list)} timeframes", flush=True)
                    timeframes = timeframes_list
                    break
            label_mapping = config['context_config']['timeframe'].get('label_mapping', {})
            ignore_other_labels = config['context_config']['timeframe'].get('ignore_other_labels', False)

        elif input_context == 'fixed_window':
            self.logger.debug(f"input_context: {input_context}")
            video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
            window_duration = config['context_config']['fixed_window']['window_duration']
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
        else:
            raise ValueError(f"Unsupported input context: {input_context}")

        if input_context == 'timeframe':
            if not isinstance(timeframes, list):
                timeframes = list(timeframes)
            if ignore_other_labels:
                timeframes = [tf for tf in timeframes if tf.get_property('label') in label_mapping]
                if not timeframes:
                    self.logger.warning("No timeframes found with labels matching the label_mapping")
                    return mmif
            print(f"DEBUG: Processing {len(timeframes)} timeframes", flush=True)
            for timeframe in timeframes:
                timeframe.add_property('timeUnit', 'milliseconds')
            print(f"DEBUG: About to extract frame numbers...", flush=True)
            all_frame_numbers = [vdh.get_representative_framenum(mmif, timeframe) for timeframe in timeframes]
            print(f"DEBUG: Extracted {len(all_frame_numbers)} frame numbers: {all_frame_numbers[:5]}...", flush=True)
            self.logger.debug(f"Extracted frame numbers: {all_frame_numbers}")
            video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
            if not video_doc:
                raise ValueError("No video document found in MMIF")
            try:
                temp_frame_numbers = all_frame_numbers.copy()
                print(f"DEBUG: About to extract {len(temp_frame_numbers)} frames from video", flush=True)
                self.logger.info(f"About to extract {len(temp_frame_numbers)} frames from video")
                import time as time_module
                extract_start = time_module.time()
                all_images = vdh.extract_frames_as_images(video_doc, temp_frame_numbers, as_PIL=True)
                extract_elapsed = time_module.time() - extract_start
                print(f"DEBUG: Frame extraction completed in {extract_elapsed:.2f} seconds, got {len(all_images)} images", flush=True)
                self.logger.info(f"Successfully extracted {len(all_images)} images in {extract_elapsed:.2f} seconds")
                if len(all_images) != len(all_frame_numbers):
                    self.logger.warning(f"Warning: Number of extracted images ({len(all_images)}) doesn't match number of frame numbers ({len(all_frame_numbers)})")
            except Exception as e:
                self.logger.error(f"Error extracting frames: {str(e)}")
                raise
            print(f"DEBUG: Starting batch processing loop, {len(timeframes)} timeframes, batch_size={batch_size}", flush=True)
            for batch_idx in tqdm.tqdm(range(0, len(timeframes), batch_size)):
                batch_timeframes = timeframes[batch_idx:batch_idx + batch_size]
                batch_images = all_images[batch_idx:batch_idx + batch_size]
                print(f"DEBUG: Processing batch {batch_idx//batch_size + 1}, {len(batch_timeframes)} timeframes", flush=True)
                self.logger.info(f"Processing batch {batch_idx//batch_size + 1}, {len(batch_timeframes)} timeframes")
                print(f"DEBUG: Preparing prompts and annotations for batch...", flush=True)
                prompts = []
                annotations_batch = []
                for idx_in_batch, timeframe in enumerate(batch_timeframes):
                    label = timeframe.get_property('label')
                    mapped_label = label_mapping.get(label, 'default')
                    prompt_tuple = self.get_prompts(mapped_label, parameters)
                    prompts.append(prompt_tuple)
                    # Get frame number for this timeframe (already extracted earlier)
                    global_idx = batch_idx + idx_in_batch
                    frame_number = all_frame_numbers[global_idx]
                    # Create a TimePoint annotation for this frame
                    timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
                    timepoint.add_property("timePoint", frame_number)
                    annotations_batch.append({
                        'source': timepoint.long_id,  # TimePoint ID
                        'document_id': video_doc.id,  # Video document
                        'origin_id': timeframe.long_id  # TimeFrame that was used
                    })
                print(f"DEBUG: Prepared {len(prompts)} prompts, about to call process_batch...", flush=True)
                self.logger.info(f"Prepared {len(prompts)} prompts, calling process_batch...")
                start_time = time.time()
                print(f"DEBUG: Calling process_batch now...", flush=True)
                process_batch(prompts, batch_images, annotations_batch)
                print(f"DEBUG: process_batch returned", flush=True)
                self.logger.info(f"process_batch completed in {time.time() - start_time:.2f} seconds")
                self.logger.debug(f"Processed batch of {len(batch_timeframes)} in {time.time() - start_time:.2f} seconds")

        elif input_context == 'fixed_window':
            prompts = []
            images_batch = []
            annotations_batch = []
            for frame_number in tqdm.tqdm(frame_numbers):
                try:
                    image = vdh.extract_frames_as_images(video_doc, [frame_number], as_PIL=True)[0]
                except:
                    self.logger.warning(f"Failed to extract frame_number: {frame_number}")
                    continue
                prompt_tuple = self.get_prompts('default', parameters)
                prompts.append(prompt_tuple)
                images_batch.append(image)
                timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
                timepoint.add_property("timePoint", frame_number)
                annotations_batch.append({
                    'source': timepoint.long_id,  # TimePoint ID
                    'document_id': video_doc.id,  # Video document
                    'origin_id': timepoint.long_id  # TimePoint that was used
                })
                if len(prompts) == batch_size:
                    start_time = time.time()
                    process_batch(prompts, images_batch, annotations_batch)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    self.logger.debug(f"Processed a batch of {batch_size} in {elapsed_time:.2f} seconds.")
                    prompts, images_batch, annotations_batch = [], [], []
            if prompts:
                start_time = time.time()
                process_batch(prompts, images_batch, annotations_batch)
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.logger.debug(f"Processed the final batch of {len(prompts)} in {elapsed_time:.2f} seconds.")
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

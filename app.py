"""
SmolVLM2 Timeframe Captioner CLAMS app.

Runs the SmolVLM2-2.2B-Instruct multimodal model over video frames sampled
from input TimeFrame annotations for prompt-driven captioning / scene
description. Each invocation processes one prompt across the labeled
TimeFrames selected by ``tfLabels``.
"""

import argparse
import logging
from typing import List

import torch
from transformers import AutoModelForImageTextToText

from clams import Restifier
from clams.app import ClamsHFPromptableApp
from mmif import Mmif, View, AnnotationTypes, DocumentTypes

from utils.timeframe import collect_timeframes_of_interest


_RAW_RESPONSE_LOG_TRUNCATE = 200


class SmolVLM2TimeframeCaptioner(ClamsHFPromptableApp):

    # The family of supported HF models (and their pinned commits) is
    # declared as ``analyzer_versions`` in metadata.py; the SDK reads
    # it from ``self.metadata`` and forwards the resolved revision to
    # ``load_hf_model(revision=...)``. This app ships a family of one
    # (only the 2.2B-Instruct variant).
    MODEL_CLS = AutoModelForImageTextToText
    DTYPE = torch.bfloat16
    PADDING_SIDE = 'left'

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug(f"Annotating with parameters: {parameters}")

        self.load_model(parameters['model'])

        prompt: List[str] = list(parameters['prompt'])
        system_prompt: str = parameters['systemPrompt']
        prompt_mode: str = parameters['promptMode']
        parallel_prompts: int = parameters['parallelPrompts']
        tflabels_of_interest: List[str] = list(parameters.get('tfLabels') or [])
        gen_params = dict(
            max_new_tokens=parameters['maxNewTokens'],
            temperature=parameters['temperature'],
            top_p=parameters['topP'],
            top_k=parameters['topK'],
        )

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)

        video_docs = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if not video_docs:
            raise ValueError(
                "SmolVLM2 Timeframe Captioner requires a VideoDocument input.")
        video_doc = video_docs[0]

        tasks = collect_timeframes_of_interest(
            mmif, new_view, video_doc, tflabels_of_interest)
        if not tasks:
            self.logger.warning(
                "No matching TimeFrames yielded any sampled frames; "
                "nothing to process.")
            return mmif

        self.logger.debug(
            f"Prepared {len(tasks)} captioning task(s) "
            f"(tfSamplingMode={parameters['tfSamplingMode']!r}):")
        for i, (task_images, tp_ids, tf_id, tf_label) in enumerate(tasks):
            self.logger.debug(
                f"  task[{i}]: TF={tf_id} (label={tf_label!r}) "
                f"-> {len(task_images)} frame(s); origin TPs: {tp_ids}")

        # Per-TF composite captioning: each task's image list is sent
        # to the model in one prompt, producing one caption per TF.
        # ``parallelPrompts`` stacks N such prompts into one forward pass.
        image_groups = [task_images for task_images, _, _, _ in tasks]
        texts: List[str] = []
        try:
            import tqdm
            iterator = tqdm.tqdm(range(0, len(image_groups), parallel_prompts))
        except ImportError:
            iterator = range(0, len(image_groups), parallel_prompts)
        for i in iterator:
            batch_groups = image_groups[i:i + parallel_prompts]
            texts.extend(
                self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    images=batch_groups,
                    prompt_mode=prompt_mode,
                    **gen_params))

        for i, ((_, tp_ids, tf_id, tf_label), text) in enumerate(
                zip(tasks, texts)):
            truncated = repr(text)
            if len(truncated) > _RAW_RESPONSE_LOG_TRUNCATE:
                truncated = truncated[:_RAW_RESPONSE_LOG_TRUNCATE] + '...'
            self.logger.debug(
                f"  task[{i}]: TF={tf_id} (label={tf_label!r}) "
                f"raw response (-{_RAW_RESPONSE_LOG_TRUNCATE} chars): {truncated}")
            self.response_to_grounded_textdocument(
                new_view, source=tf_id, response=text.strip(),
                origins=tp_ids, origination='derived')

        return mmif


def get_app():
    return SmolVLM2TimeframeCaptioner()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000",
                        help="set port to listen")
    parser.add_argument("--production", action="store_true",
                        help="run gunicorn server")
    parsed_args = parser.parse_args()

    app = get_app()
    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()

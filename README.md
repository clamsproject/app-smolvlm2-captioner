# SmolVLM2 Captioner CLAMS App

This CLAMS app integrates the [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) multimodal model for describing video frames. 

SmolVLM2 is a compact yet powerful multimodal language model designed for efficiency. It can process images directly while providing detailed descriptions, making it ideal for video frame analysis and captioning.

## Features

- Direct analysis of images and video frames with SmolVLM2-2.2B-Instruct multimodal model
- Custom prompting based on frame type (e.g., slates vs. content frames)
- Multiple processing modes:
  - Fixed window: sample frames at regular intervals
  - Timeframe: process frames from specific timeframes
  - Image: process individual images

## Installation

### Using Docker (Recommended)

```bash
docker pull clamsproject/app-smolvlm2-captioner
docker run -p 5000:5000 clamsproject/app-smolvlm2-captioner
```

### From Source

```bash
git clone https://github.com/clamsproject/app-smolvlm2-captioner.git
cd app-smolvlm2-captioner
pip install -r requirements.txt
python app.py
```

## Usage

The app can be used with the CLAMS workflow manager or standalone via REST API.

### Running with a config file

You can run the app using a YAML config file to specify prompts and context configuration. For example:

```bash
python cli.py --config config/default.yaml input.mmif output.mmif
```

- `--config` specifies the path to the YAML config file.
- The config file can define `default_prompt`, `custom_prompts`, and `context_config`.
- You can override `defaultPrompt` and `promptMap` via CLI if needed.

### Using custom configs with containers

When running the containerized version, you can override the built-in configuration directory by mounting an external directory to `/app/config` in the container. This allows you to use completely custom configuration files without rebuilding the container image. For example: `docker run -v /path/to/custom/configs:/app/config app-smolvlm2-captioner python cli.py --config custom.yaml input.mmif output.mmif`

### Prompt Handling
- The app will use a label-specific prompt from `custom_prompts` if available (via `promptMap`), otherwise it will use the `default_prompt` (via `defaultPrompt`).
- If neither is provided, a generic fallback prompt will be used.

See `config/default.yaml` for an example config file format.

### Configuration

The app uses YAML configuration files to control behavior. Sample configuration files are provided in the `config/` directory. The main parameters include:

- `default_prompt`: The prompt template to use for standard frames
- `custom_prompts`: Specialized prompts for different frame types
- `context_config`: Control how the app processes input (fixed_window, timeframe, image)

### Example Prompts

Default prompt format:
```
I'm looking at a video frame. Can you describe what is shown in this frame? Include any important details about people, objects, text, and setting visible in the frame.
```

Special slate prompt:
```
This is a slate frame from a video. Please analyze it and extract all key information: 
- Title of the program
- Date of recording
- Any identifiers, codes, or numbers
- Name of production company or network
- Any other textual information visible 

Format the information clearly.
```

## Output Format

When processing in `timeframe` mode, the app produces `TextDocument` and `Alignment` annotations with specific linking conventions:

### Annotation Linking

For each processed frame, the app creates:

1. **TextDocument**: Contains the captioned text with:
   - `origin`: Points to the **TimeFrame** annotation that was the source of the processing task
   - `document`: Points to the video document

2. **Alignment**: Links the TextDocument to its source with:
   - `source`: Points to the **TimePoint** annotation that was actually processed (the specific frame)
   - `target`: Points to the TextDocument created

### Example Output

```json
{
  "@type": "http://mmif.clams.ai/vocabulary/TextDocument/v1",
  "properties": {
    "document": "m1",
    "origin": "v_1:tf_38",
    "provenance": "derived",
    "mime": "application/json",
    "text": { "@value": "JOHN MARCUM Africa", "@language": "en" },
    "id": "v_2:td_9"
  }
},
{
  "@type": "http://mmif.clams.ai/vocabulary/Alignment/v1",
  "properties": {
    "source": "v_0:tp_5389",
    "target": "v_2:td_9",
    "id": "v_2:al_9"
  }
}
```

In this example:
- The TextDocument's `origin` (`v_1:tf_38`) references the TimeFrame that triggered processing
- The Alignment's `source` (`v_0:tp_5389`) references the specific TimePoint (frame) that was captioned
- This allows downstream applications to trace both which TimeFrame was processed AND which specific frame within that TimeFrame was used

### Processing Multiple Representatives

By default, only the first representative TimePoint in each TimeFrame is processed. You can enable processing of all representatives either globally or per-label.

**Per-label configuration (recommended):**
```yaml
context_config:
  timeframe:
    all_representatives:
      slate: true      # Process all representatives for slates
      chyron: false    # Only first representative for chyrons
```

**Global default via CLI parameter:**
```bash
python cli.py --allRepresentatives input.mmif output.mmif
```

The per-label config takes precedence over the CLI parameter. When enabled for a label, each representative TimePoint in the TimeFrame will be captioned separately, with each resulting TextDocument linked to its corresponding TimePoint via the Alignment annotation.

### Fallback Behavior

If a TimeFrame does not have a `representatives` property (no specific TimePoint to reference), the app falls back to using the middle frame of the TimeFrame. In this case, the Alignment's `source` will point to the TimeFrame itself since there is no TimePoint to reference.

## Model Information

- SmolVLM2-2.2B-Instruct is a lightweight multimodal model that can process both text and images
- The model is quantized to 4-bit for efficiency
- Developed by HuggingFaceTB
- Model link: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct

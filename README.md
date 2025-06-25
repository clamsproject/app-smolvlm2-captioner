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

## Model Information

- SmolVLM2-2.2B-Instruct is a lightweight multimodal model that can process both text and images
- The model is quantized to 4-bit for efficiency
- Developed by HuggingFaceTB
- Model link: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct

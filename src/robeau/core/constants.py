from src.config.settings import PROJECT_DIR_PATH
import os

# Robeau specific paths
ROBEAU_PROMPTS_JSON_FILE_PATH = os.path.join(
    PROJECT_DIR_PATH, "src/robeau/jsons/processed_for_robeau/robeau_prompts.json"
)
AUDIO_MAPPINGS_FILE_PATH = os.path.join(
    PROJECT_DIR_PATH, "src/robeau/jsons/processed_for_robeau/robeau_responses.json"
)

USER_PROMPT_LABELS = ["Prompt", "Whisper", "Plea", "Answer", "Greeting"]
ROBEAU_LABELS = ["Response", "Question", "Test"]
SYSTEM_LABELS = ["Input", "Output", "LogicGate", "TrafficGate"]

from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, CLIPModel
import robomimic.utils.lang_utils as LangUtils
import torch
import pdb


def get_text_model(task_name, language_emb_model):
    if language_emb_model == "clip":
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    elif language_emb_model in {"robomimic", "lang_encoder"}:
        tokenizer = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_model = LangUtils.LangEncoder(device=device)
    else:
        tokenizer = None
        text_model = None

    if "libero_10" in task_name:
        max_length = 30
    elif "umi" in task_name:
        max_length = 30
    else:
        max_length = 30

    return text_model, tokenizer, max_length


def extract_text_features(text_model, text_tokens, language_emb_model):
    with torch.no_grad():
        if language_emb_model == "clip":
            text_latents = text_model.get_text_features(**text_tokens)
        elif language_emb_model in {"robomimic", "lang_encoder"}:
            if isinstance(text_tokens, str):
                text_tokens = [text_tokens]
            text_latents = text_model.get_lang_emb(text_tokens)

        else:
            pdb.set_trace()

    return text_latents

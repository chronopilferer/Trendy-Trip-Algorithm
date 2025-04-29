import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from utils.file_io import save_result, copy_image
from utils.constants import SCENE_THRESHOLD, OBJECT_THRESHOLD, MARGIN_DELTA

def load_clip_model(name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc   = CLIPProcessor.from_pretrained(name)
    mdl    = CLIPModel.from_pretrained(name).to(device).eval()
    return proc, mdl, device

def compute_similarity(
    img: Image.Image,
    texts: list,
    proc: CLIPProcessor,
    mdl: CLIPModel,
    device: torch.device
) -> np.ndarray:
    inputs = proc(text=texts, images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_emb = mdl.get_image_features(pixel_values=inputs.pixel_values)
        txt_emb = mdl.get_text_features(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return (img_emb @ txt_emb.T).squeeze(0).cpu().numpy()

def classify_scene_object(
    path: str,
    prompts,
    proc: CLIPProcessor,
    mdl: CLIPModel,
    device: torch.device
) -> tuple:
    img = Image.open(path).convert("RGB")
    s_score = float(np.max(compute_similarity(img, prompts["scene"],  proc, mdl, device)))
    o_score = float(np.max(compute_similarity(img, prompts["object"], proc, mdl, device)))
    return s_score, o_score

def process_clip_filtering(
    json_dir: str,
    out_dir: str,
    model_name: str,
    prompts
):
    os.makedirs(out_dir, exist_ok=True)
    for d in ("pass", "hold", "non-pass"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    proc, mdl, device = load_clip_model(model_name)

    for fn in os.listdir(json_dir):
        if not fn.lower().endswith(".json"):
            continue

        json_path = os.path.join(json_dir, fn)
        rec = json.load(open(json_path, "r", encoding="utf-8"))

        if not rec.get("pass", False):
            continue

        img_path = rec.get("filepath")
        if not img_path or not os.path.isfile(img_path):
            continue

        s_score, o_score = classify_scene_object(img_path, prompts, proc, mdl, device)
        rec.update({"scene_score": s_score, "object_score": o_score})

        print("scene_score", s_score, "object_score", o_score)

        if s_score <= o_score or o_score >= OBJECT_THRESHOLD:
            decision = "non-pass"
        elif s_score - MARGIN_DELTA >= o_score and s_score >= SCENE_THRESHOLD:
            decision = "pass"
        else:
            decision = "hold"

        rec["clip_decision"] = decision
        rec["pass"] = True if decision in ("pass", "hold") else False

        save_result(rec, json_path)

        copy_image(img_path, out_dir, decision)

        print(f"[{decision.upper()}] {fn} → scene={s_score:.2f}, obj={o_score:.2f}")

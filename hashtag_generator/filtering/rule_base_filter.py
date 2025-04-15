from utils.image_utils import read_image_unicode_safe, is_too_dark_or_bright, is_low_resolution, is_low_entropy

def image_filter_analysis(image_path: str) -> dict:
    try:
        image = read_image_unicode_safe(image_path)
    except Exception as e:
        return {"status": "error", "reason": f"file_read_failed: {str(e)}"}

    light_cond = is_too_dark_or_bright(image)
    low_res_flag = is_low_resolution(image)
    entropy_flag, entropy_score = is_low_entropy(image)

    result = {
        "status": "pass",
        "light": str(light_cond),
        "low_resolution": bool(low_res_flag),
        "low_entropy": bool(entropy_flag),
        "entropy_score": float(entropy_score),
    }

    if light_cond != "ok":
        result["status"] = "fail"
        result["reason"] = str(light_cond)
    elif low_res_flag:
        result["status"] = "fail"
        result["reason"] = "low_resolution"
    elif entropy_flag:
        result["status"] = "fail"
        result["reason"] = "low_entropy"

    return result

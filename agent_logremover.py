import os
import re
import sys
import warnings
import logging
from typing import Dict, Any

# Reduce noisy warnings from transformers/huggingface_hub during local runs
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings(
    "ignore",
    message="The 'model_id' parameter will be required in version 2.0.0.*",
)
warnings.filterwarnings(
    "ignore",
    message="The attention mask is not set and cannot be inferred from input because pad token is same as eos token.*",
)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

from smolagents import CodeAgent, Tool, TransformersModel


# ---------- Pure Python log cleaner ----------

def remove_log_pattern_py(file_path: str, pattern: str) -> str:
    """
    Remove lines matching a regex pattern from a logfile.
    Returns path of cleaned file and basic stats.
    """
    if not os.path.exists(file_path):
        return f"Error: file not found: {file_path}"

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Defensive fix: some agents produce a negative-lookahead pattern like
    # '^(?!.*stdin.*)' which matches lines that DO NOT contain 'stdin'. If
    # such a pattern is used with our semantics (remove matching lines), it
    # would remove the wrong set of lines (keep only stdin lines). Detect
    # common negative-lookahead forms and convert them to the positive
    # pattern matching the inner substring.
    neg_m = re.match(r"^\^?\(\?\!\.\*(?P<inner>.+?)\.\*\)\s*$", pattern)
    if neg_m:
        inner = neg_m.group("inner")
        # use a word-boundary aware pattern for safety
        pattern = rf"\b{re.escape(inner)}\b"

    cleaned_lines = [line for line in lines if not re.search(pattern, line)]
    output_path = file_path.rsplit(".", 1)[0] + "_out.log"

    # Remove legacy files from older runs to keep only a single up-to-date output
    base = file_path.rsplit(".", 1)[0]
    for legacy in (base + "_cleaned.log", base + "_deduped.log"):
        try:
            if os.path.exists(legacy):
                os.remove(legacy)
        except Exception:
            pass

    # Ensure no backups are kept: remove any existing _out.log before writing
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception:
        pass

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    return (
        f"Cleaned file saved to {output_path}. "
        f"Kept {len(cleaned_lines)} lines out of {len(lines)}."
    )


# ---------- Smolagents Tool wrapper ----------

class RemoveLogPatternTool(Tool):
    name = "remove_log_pattern"
    description = (
        "Remove all lines that match a given regular expression pattern "
        "from a logfile on disk."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the logfile to clean",
        },
        "pattern": {
            "type": "string",
            "description": "Regex pattern describing lines to remove",
        },
    }
    output_type = "string"

    # smolagents expects the `forward` method signature to match the keys
    # declared in `inputs` (after `self`). Define parameters accordingly.
    def forward(self, file_path: str, pattern: str) -> str:
        if not file_path:
            raise ValueError("RemoveLogPatternTool requires a 'file_path' input")
        return remove_log_pattern_py(file_path, pattern)


def build_agent() -> CodeAgent:
    # Use a tiny text-only model that works with AutoModelForCausalLM.
    # gpt2 is easy to run on CPU and supported out of the box.[web:67]
    model = TransformersModel(
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        max_new_tokens=128,
        trust_remote_code=False,
    )

    # Ensure tokenizer has a distinct pad token so attention_mask can be inferred
    # by the transformers utilities. If we add a pad token we must resize the
    # model embeddings to match the tokenizer vocabulary size.
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
        pad_token = getattr(tokenizer, "pad_token", None)
        eos_token = getattr(tokenizer, "eos_token", None)
        if pad_token is None or pad_token == eos_token:
            # Ensure pad token exists in the vocabulary and is distinct from eos_token
            # Add token to tokenizer and explicitly set pad_token, then resize embeddings.
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            tokenizer.pad_token = "<pad>"
            if hasattr(model, "model") and hasattr(model.model, "resize_token_embeddings"):
                try:
                    # Prefer to pass mean_resizing=False to avoid the informational message
                    model.model.resize_token_embeddings(len(tokenizer), mean_resizing=False)  # type: ignore[arg-type]
                except TypeError:
                    # Older transformers versions may not accept mean_resizing; fall back
                    try:
                        model.model.resize_token_embeddings(len(tokenizer))
                    except Exception:
                        pass
                except Exception:
                    # Some model classes may still fail resizing; ignore and proceed
                    pass

    tools = [RemoveLogPatternTool()]

    system_prompt = (
        "You are a log-cleaning assistant. "
        "The user will describe what pattern to remove from which logfile. "
        "Infer a simple regex for the pattern the user describes and call "
        "remove_log_pattern with the correct file_path and pattern. "
        "After calling the tool, briefly explain what you removed."
    )

    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=3,
        verbosity_level=0,
    )
    # smolagents uses a read-only `system_prompt` property; update prompt
    # templates dictionary instead.
    agent.prompt_templates["system_prompt"] = system_prompt
    return agent


# simplified sanitize helper

def sanitize_agent_output(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"</?code>", "", text)
    m = re.search(r"```(?:python|regex)?\s*([\s\S]*?)\s*```", text)
    if m:
        code = m.group(1).strip()
        if re.search(r"\\b|\\w|\^|\\.|\*|\(|\[", code):
            return f"Suggested regex: {code}"
        return code.splitlines()[0]

    text = re.sub(r"Explanation:.*", "", text, flags=re.S)
    text = " ".join(text.split())
    return text.split('. ')[0].strip()


def _normalize_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    m_drive = re.match(r"^/([a-zA-Z])/(.+)", p)
    if m_drive:
        drive, rest = m_drive.group(1).upper(), m_drive.group(2)
        p = f"{drive}:/" + rest
    p = p.replace("/", os.sep).replace("\\\\", os.sep)
    return p


def deduplicate_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"Error: file not found: {file_path}"
    seen = set()
    kept = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line in seen:
                continue
            seen.add(line)
            kept.append(line)

    out_path = file_path.rsplit(".", 1)[0] + "_out.log"

    # Remove legacy files from older runs to keep only a single up-to-date output
    base = file_path.rsplit(".", 1)[0]
    for legacy in (base + "_cleaned.log", base + "_deduped.log"):
        try:
            if os.path.exists(legacy):
                os.remove(legacy)
        except Exception:
            pass

    # Ensure no backups are kept: remove any existing _out.log before writing
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception:
        pass

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(kept)
    original_count = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore'))
    removed = original_count - len(kept)
    return f"Deduplicated {removed} lines from {file_path} (saved: {out_path})"


def process_agent_response_fixed(user_input: str, result_str: str) -> str:
    # If result already looks like a tool response, return it
    if result_str.startswith("Removed") or result_str.startswith("Error:"):
        return result_str

    # handle dedup explicit instruction
    dedup_keywords = re.search(r"\b(dedupe|deduplicate|duplicate|duplicates|duplicated|remove repeated|repeating lines|keep only one|keep one occurrence|unique lines)\b", user_input, re.I)
    if dedup_keywords:
        # explicit phrasing first
        m_path = re.search(r'remove all lines containing\s+"?([^"\s]+)"?\s+from\s+(.+)', user_input, re.I)
        path = m_path.group(2).strip() if m_path else None
        if not path:
            # look for existing file tokens mentioned in the input
            tokens = re.findall(r'[^"\s]+', user_input)
            for tok in tokens:
                candidate = _normalize_path(tok)
                if os.path.exists(candidate):
                    path = candidate
                    break
        if path:
            return deduplicate_file(path)

    # otherwise, try to extract a suggested regex from agent output
    code_block = re.search(r"```(?:python|regex)?\s*([\s\S]*?)\s*```", result_str)
    candidate = code_block.group(1).strip() if code_block else None
    if not candidate:
        m = re.search(r"Suggested regex:\s*(.+)", result_str)
        if m:
            candidate = m.group(1).strip()

    if candidate:
        # find a path in the user input or the agent output
        m_path = re.search(r'remove all lines containing\s+"?([^"\s]+)"?\s+from\s+(.+)', user_input, re.I)
        path = m_path.group(2).strip() if m_path else None
        if not path:
            tokens = re.findall(r'[^"\s]+', result_str)
            for tok in tokens:
                candidate = _normalize_path(tok)
                if os.path.exists(candidate):
                    path = candidate
                    break
        if path:
            return remove_log_pattern_py(_normalize_path(path), candidate)

    return sanitize_agent_output(result_str)


if __name__ == "__main__":
    # Build agent while suppressing noisy build-time messages from transformers
    def safe_build_agent():
        import io
        import contextlib

        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            agent = build_agent()
        noisy = ["attention mask is not set", "The attention mask is not set and cannot be inferred"]
        for buf in (out_buf.getvalue(), err_buf.getvalue()):
            for line in buf.splitlines():
                if any(p.lower() in line.lower() for p in noisy):
                    continue
        return agent

    agent = safe_build_agent()

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:]).strip()
    else:
        user_input = sys.stdin.read().strip()

    output_path = os.path.join(os.getcwd(), "agent_output.txt")
    try:
        if not user_input:
            with open(output_path, "w", encoding="utf-8") as out:
                out.write("Error: no instruction provided.\n")
            sys.exit(1)

        result = safe_agent_run(agent, user_input)
        result_str = str(result)
        final_out = process_agent_response_fixed(user_input, result_str)
        if not final_out.startswith(("Removed", "Error:")):
            final_out = sanitize_agent_output(final_out)
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(final_out)
    except Exception as e:
        err_text = str(e)
        recovered = process_agent_response_fixed(user_input, err_text)
        final_out = recovered if recovered.startswith(("Removed", "Error:")) else sanitize_agent_output(recovered)
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(final_out)
        sys.exit(0)
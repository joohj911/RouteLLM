"""
BFCL에서 Qwen3.5-2B / Qwen3.5-9B를 평가하여 eval_results.json 을 생성하는 스크립트.

prepare_bfcl_data.py embed 이후, prepare_bfcl_data.py convert 이전에 실행.

사용법:
  python eval_bfcl_models.py \\
    --prompts-path ./bfcl_data/prompts.json \\
    --output-path ./eval_results.json \\
    --weak-model Qwen/Qwen3.5-2B \\
    --strong-model Qwen/Qwen3.5-9B

옵션:
  --load-in-4bit   : 4-bit 양자화 (VRAM 부족 시)
  --max-new-tokens : 생성 최대 토큰 수 (기본 512)
  --device         : cuda / cpu (기본 cuda)

BFCL 카테고리별 평가 기준:
  relevance/simple/multiple/parallel : ground truth 함수 호출 일치 여부
  irrelevance                         : 함수 호출 없음(거부)이 정답
  multi_turn                          : 첫 번째 턴 기준 평가
"""

import argparse
import json
import re
import urllib.request

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# BFCL v4 데이터는 Gorilla GitHub 레포에서 직접 다운로드.
GORILLA_RAW_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main"
    "/berkeley-function-call-leaderboard/bfcl_eval/data"
)

# BFCL v4 split 이름. prepare_bfcl_data.py의 BFCL_SPLITS와 동일하게 유지.
# 파일 목록 출처: gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/
# 제외: BFCL_v4_memory (메모리 백엔드 필요), BFCL_v4_web_search (웹 검색 API 필요),
#        BFCL_v4_format_sensitivity (비채점)
BFCL_SPLITS = [
    # Non-live
    "BFCL_v4_simple_python",
    "BFCL_v4_simple_java",
    "BFCL_v4_simple_javascript",
    "BFCL_v4_multiple",
    "BFCL_v4_parallel",
    "BFCL_v4_parallel_multiple",
    "BFCL_v4_irrelevance",
    # Live
    "BFCL_v4_live_simple",
    "BFCL_v4_live_multiple",
    "BFCL_v4_live_parallel",
    "BFCL_v4_live_parallel_multiple",
    "BFCL_v4_live_relevance",
    "BFCL_v4_live_irrelevance",
    # Multi-turn
    "BFCL_v4_multi_turn_base",
    "BFCL_v4_multi_turn_miss_func",
    "BFCL_v4_multi_turn_miss_param",
    "BFCL_v4_multi_turn_long_context",
]


# ─────────────────────────────────────────────
# 모델 로드 / 언로드
# ─────────────────────────────────────────────

def load_model(model_name: str, device: str, load_in_4bit: bool):
    """
    device 형식:
      "cuda:0", "cuda:1"  → 해당 GPU에만 올림 (device_map={"": device})
      "cuda"              → 가용 GPU 전체에 자동 분산 (device_map="auto")
      "cpu"               → CPU
    """
    print(f"\nLoading {model_name} on {device} (4-bit={load_in_4bit}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # H100은 bfloat16이 float16보다 빠르고 수치적으로 안정적
    kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": device} if ":" in device else "auto"
    elif device == "cpu":
        kwargs["device_map"] = None
    elif ":" in device:
        # "cuda:0" / "cuda:1" → 단일 GPU에 고정
        kwargs["device_map"] = {"": device}
    else:
        # "cuda" → 가용 GPU 자동 분산
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def unload_model(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# BFCL 데이터 로드
# ─────────────────────────────────────────────

def _fetch_split(split: str) -> list[dict]:
    """GitHub raw URL에서 BFCL split JSON을 다운로드하여 반환."""
    url = f"{GORILLA_RAW_BASE}/{split}.json"
    with urllib.request.urlopen(url) as resp:
        content = resp.read().decode("utf-8")
    data = json.loads(content)
    if isinstance(data, list):
        return data
    return [json.loads(line) for line in content.strip().splitlines() if line.strip()]


def load_bfcl_by_id() -> dict:
    """모든 BFCL split을 로드하여 id → sample 딕셔너리로 반환."""
    id_to_sample = {}
    for split in BFCL_SPLITS:
        try:
            samples = _fetch_split(split)
        except Exception as e:
            print(f"  [skip] {split}: {e}")
            continue
        for sample in samples:
            id_to_sample[sample["id"]] = sample
    print(f"Loaded {len(id_to_sample)} BFCL samples total.")
    return id_to_sample


# ─────────────────────────────────────────────
# 프롬프트 포맷
# ─────────────────────────────────────────────

def build_tools(function_list: list) -> list:
    """BFCL function 리스트를 OpenAI 호환 tool 형식으로 변환."""
    tools = []
    for func in function_list:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        )
    return tools


def build_messages(question: list, is_multi_turn: bool) -> list:
    """
    BFCL question 필드에서 첫 번째 턴 메시지를 추출한다.

    single-turn: question = [[{role, content}, ...]]
    multi-turn:  question = [[turn1_msgs], [turn2_msgs], ...]
    → 모두 첫 번째 턴만 사용 (multi-turn은 turn1 기준 평가)
    """
    if not question:
        return []
    first_turn = question[0] if isinstance(question[0], list) else question
    return [{"role": m["role"], "content": m["content"]} for m in first_turn]


# ─────────────────────────────────────────────
# 추론
# ─────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    messages: list,
    tools: list,
    max_new_tokens: int,
    device: str,
) -> str:
    """Qwen3.5 chat template을 적용하여 응답을 생성한다."""
    apply_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if tools:
        apply_kwargs["tools"] = tools
    # Qwen3 계열: thinking 비활성화로 clean output
    try:
        apply_kwargs["enable_thinking"] = False
    except Exception:
        pass

    text = tokenizer.apply_chat_template(messages, **apply_kwargs)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─────────────────────────────────────────────
# 출력 파싱
# ─────────────────────────────────────────────

def parse_tool_calls(response: str) -> list[dict]:
    """
    모델 응답에서 tool call JSON을 추출한다.

    Qwen3.5 출력 형식 예시:
      <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    또는 JSON 블록으로 직접 출력될 수도 있음.
    """
    tool_calls = []

    # <tool_call>...</tool_call> 블록 파싱
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    for match in pattern.finditer(response):
        try:
            obj = json.loads(match.group(1).strip())
            tool_calls.append(obj)
        except json.JSONDecodeError:
            pass

    if tool_calls:
        return tool_calls

    # fallback: 응답 전체가 JSON 배열/객체인 경우
    stripped = response.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                tool_calls = [parsed]
            elif isinstance(parsed, list):
                tool_calls = parsed
        except json.JSONDecodeError:
            pass

    return tool_calls


# ─────────────────────────────────────────────
# 정답 비교
# ─────────────────────────────────────────────

def normalize(v) -> str:
    """비교를 위한 값 정규화: 소문자 변환, 공백 제거."""
    return str(v).lower().strip()


def calls_match(predicted: dict, expected: dict) -> bool:
    """
    함수명 일치 + expected의 모든 파라미터가 predicted에 있고 값이 일치하는지 확인.
    값 비교는 string 기준 (타입 변환 포함).
    """
    if normalize(predicted.get("name", "")) != normalize(expected.get("name", "")):
        return False

    pred_args = predicted.get("arguments", {})
    exp_args = expected.get("arguments", {})

    for key, exp_val in exp_args.items():
        if key not in pred_args:
            return False
        if normalize(pred_args[key]) != normalize(exp_val):
            # 숫자 비교 재시도
            try:
                if float(pred_args[key]) != float(exp_val):
                    return False
            except (ValueError, TypeError):
                return False
    return True


def is_pass(predicted_calls: list[dict], ground_truth, is_irrelevance: bool) -> bool:
    """
    BFCL 정답과 모델 출력을 비교하여 pass/fail 반환.

    ground_truth: list of {name, arguments} 또는 빈 리스트 (irrelevance)
    is_irrelevance: True이면 함수 호출 없음이 정답
    """
    if is_irrelevance:
        # irrelevance: 함수 호출을 하지 않아야 정답
        return len(predicted_calls) == 0

    if not ground_truth:
        return len(predicted_calls) == 0

    # ground_truth가 문자열인 경우 파싱 시도
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return False

    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]

    if len(predicted_calls) < len(ground_truth):
        return False

    # 각 ground truth call이 predicted에 있는지 순서 무관하게 확인
    matched = [False] * len(predicted_calls)
    for exp_call in ground_truth:
        found = False
        for i, pred_call in enumerate(predicted_calls):
            if not matched[i] and calls_match(pred_call, exp_call):
                matched[i] = True
                found = True
                break
        if not found:
            return False
    return True


# ─────────────────────────────────────────────
# 단일 샘플 평가
# ─────────────────────────────────────────────

def evaluate_sample(model, tokenizer, sample: dict, max_new_tokens: int, device: str) -> bool:
    split_name = sample.get("_split", "")
    is_irrelevance = "irrelevance" in split_name
    is_multi_turn = "multi_turn" in split_name

    tools = build_tools(sample.get("function", []))
    messages = build_messages(sample.get("question", []), is_multi_turn)

    if not messages:
        return False

    response = run_inference(model, tokenizer, messages, tools, max_new_tokens, device)
    predicted_calls = parse_tool_calls(response)

    ground_truth = sample.get("ground_truth", [])
    # multi_turn: ground_truth가 중첩 리스트인 경우 첫 턴만 사용
    if is_multi_turn and ground_truth and isinstance(ground_truth[0], list):
        ground_truth = ground_truth[0]

    return is_pass(predicted_calls, ground_truth, is_irrelevance)


# ─────────────────────────────────────────────
# 전체 평가 실행
# ─────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    prompts: list[dict],
    id_to_sample: dict,
    max_new_tokens: int,
    device: str,
    load_in_4bit: bool,
) -> tuple[dict[str, bool], str]:
    """한 모델을 전체 BFCL 샘플에 대해 평가하고 {id: pass} 딕셔너리 반환."""
    model, tokenizer = load_model(model_name, device, load_in_4bit)

    results = {}
    failed_samples = 0
    col_name = f"{model_name.split('/')[-1].lower()}_pass"

    for prompt_meta in tqdm(prompts, desc=model_name.split("/")[-1]):
        sample_id = prompt_meta["id"]
        sample = id_to_sample.get(sample_id)
        if sample is None:
            results[sample_id] = False
            failed_samples += 1
            continue

        sample["_split"] = prompt_meta.get("bfcl_split", "")

        try:
            passed = evaluate_sample(model, tokenizer, sample, max_new_tokens, device)
        except Exception as e:
            print(f"\n  [error] {sample_id}: {e}")
            passed = False
            failed_samples += 1

        results[sample_id] = passed

    unload_model(model)

    n_pass = sum(results.values())
    print(f"\n{model_name.split('/')[-1]} results:")
    print(f"  Pass: {n_pass}/{len(results)} ({n_pass/max(len(results),1)*100:.1f}%)")
    if failed_samples:
        print(f"  Errors/missing: {failed_samples}")

    return results, col_name


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BFCL에서 두 모델을 평가하여 eval_results.json 생성"
    )
    parser.add_argument(
        "--prompts-path",
        type=str,
        required=True,
        help="prepare_bfcl_data.py embed 이 생성한 prompts.json 경로",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./eval_results.json",
        help="결과 저장 경로 (기본값: ./eval_results.json)",
    )
    parser.add_argument(
        "--weak-model",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="약한 모델 HuggingFace ID",
    )
    parser.add_argument(
        "--strong-model",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="강한 모델 HuggingFace ID",
    )
    parser.add_argument(
        "--weak-device",
        type=str,
        default="cuda:0",
        help="weak 모델을 올릴 GPU (기본값: cuda:0)",
    )
    parser.add_argument(
        "--strong-device",
        type=str,
        default="cuda:1",
        help="strong 모델을 올릴 GPU (기본값: cuda:1)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="4-bit 양자화 (VRAM 부족 시 사용, bitsandbytes 필요)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    # prompts.json 로드
    with open(args.prompts_path) as f:
        prompts = json.load(f)
    print(f"Prompts to evaluate: {len(prompts)}")

    # BFCL 전체 데이터 로드 (function 정의 + ground_truth 포함)
    print("\nLoading BFCL dataset from GitHub ...")
    id_to_sample = load_bfcl_by_id()

    # weak 모델 평가 (--weak-device, 기본 cuda:0)
    weak_results, weak_col = evaluate_model(
        args.weak_model, prompts, id_to_sample,
        args.max_new_tokens, args.weak_device, args.load_in_4bit,
    )

    # strong 모델 평가 (--strong-device, 기본 cuda:1); weak 언로드 후 실행
    strong_results, strong_col = evaluate_model(
        args.strong_model, prompts, id_to_sample,
        args.max_new_tokens, args.strong_device, args.load_in_4bit,
    )

    # eval_results.json 생성
    output = []
    for prompt_meta in prompts:
        sid = prompt_meta["id"]
        # key는 모델 short name (e.g. "qwen3.5-2b_pass")
        # prepare_bfcl_data.py convert의 weak_key/strong_key와 맞춰야 함
        weak_short = args.weak_model.split("/")[-1].lower()
        strong_short = args.strong_model.split("/")[-1].lower()
        output.append(
            {
                "id": sid,
                f"{weak_short}_pass": weak_results.get(sid, False),
                f"{strong_short}_pass": strong_results.get(sid, False),
            }
        )

    with open(args.output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved eval_results → {args.output_path}  ({len(output)} samples)")
    print(
        "\n[다음 단계] prepare_bfcl_data.py convert 실행 시 "
        f"--weak-model {weak_short} --strong-model {strong_short} 로 지정하세요."
    )

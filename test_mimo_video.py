"""
Step 0: 验证 Mimo v2.5 视频理解能力
测试三个关键问题：
1. 能否接受 base64 编码的视频输入？
2. 能否识别篮球进球事件？
3. 能否返回进球发生的时间戳？
"""
import os
import base64
import json
import time
from pathlib import Path
from openai import OpenAI

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5"
VIDEO_PATH = Path(__file__).parent / "video.mp4"


def estimate_video_tokens(file_size_mb: int, duration_s: int, fps: float = 2) -> dict:
    """粗略估算 token 消耗"""
    # 基于 Mimo 文档的 estimate_video_tokens 逻辑简化
    frames = int(duration_s * fps)
    frames = min(frames, 2048)
    # 视频 token 粗略估算: ~500-1500 tokens/frame 取决于分辨率
    visual_tokens_low = frames * 500
    visual_tokens_high = frames * 1500
    # 音频 token: ~6.25/s
    audio_tokens = int(duration_s * 6.25)
    return {
        "file_size_mb": file_size_mb,
        "duration_s": duration_s,
        "fps": fps,
        "estimated_frames": frames,
        "visual_tokens_range": f"{visual_tokens_low:,} - {visual_tokens_high:,}",
        "audio_tokens": f"{audio_tokens:,}",
        "total_low": f"{visual_tokens_low + audio_tokens:,}",
        "total_high": f"{visual_tokens_high + audio_tokens:,}",
    }


def encode_video_base64(path: Path) -> tuple[str, int]:
    """base64 编码视频，返回 (data_url, original_size_mb)"""
    with open(path, "rb") as f:
        data = f.read()
    size_mb = len(data) / (1024 * 1024)
    b64 = base64.b64encode(data).decode("utf-8")
    data_url = f"data:video/mp4;base64,{b64}"
    b64_size_mb = len(data_url) / (1024 * 1024)
    print(f"  原始大小: {size_mb:.1f} MB")
    print(f"  Base64 编码后: {b64_size_mb:.1f} MB")
    if b64_size_mb > 50:
        print(f"  ⚠️ 警告: 超过 Mimo base64 限制 (50MB)!")
    return data_url, size_mb


def get_video_duration(path: Path) -> int:
    """用 ffprobe 获取视频时长（秒）"""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True
        )
        return int(float(result.stdout.strip()))
    except Exception:
        return 0


def test_video_understanding(client: OpenAI, data_url: str, prompt: str,
                            fps: float = 2, max_tokens: int = 4096) -> dict:
    """发送视频到 Mimo v2.5 并返回结果"""
    start = time.time()
    response = client.chat.completions.create(
        model=MIMO_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": data_url},
                        "fps": fps,
                        "media_resolution": "default",
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content

    # 提取 token 使用信息
    usage = {}
    if hasattr(response, "usage") and response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "elapsed_s": round(elapsed, 1),
        "content": content,
        "usage": usage,
        "model": response.model,
    }


def main():
    if not os.environ.get("MIMO_API_KEY"):
        print("❌ 请设置环境变量 MIMO_API_KEY")
        return

    if not VIDEO_PATH.exists():
        print(f"❌ 找不到视频文件: {VIDEO_PATH}")
        return

    client = OpenAI(
        api_key=os.environ["MIMO_API_KEY"],
        base_url=MIMO_BASE_URL,
    )

    # 获取视频信息
    duration_s = get_video_duration(VIDEO_PATH)

    print("=" * 60)
    print("🧪 Mimo v2.5 视频理解能力验证")
    print("=" * 60)

    # 估算 token
    file_size_mb = VIDEO_PATH.stat().st_size / (1024 * 1024)
    est = estimate_video_tokens(file_size_mb, duration_s, fps=2)
    print(f"\n📊 Token 估算 (fps=2, {duration_s}s 视频):")
    for k, v in est.items():
        print(f"  {k}: {v}")

    # 编码视频
    print(f"\n📦 编码视频...")
    data_url, _ = encode_video_base64(VIDEO_PATH)

    # --- 测试 1: 通用描述 ---
    print(f"\n{'─' * 60}")
    print("📋 测试 1: 通用视频理解")
    print("   Prompt: '请用中文简要描述这段视频的内容'")
    print(f"{'─' * 60}")

    result1 = test_video_understanding(
        client, data_url,
        "请用中文简要描述这段视频的内容，包括场景、人物活动等。"
    )
    print(f"  ⏱️ 耗时: {result1['elapsed_s']}s")
    print(f"  🪙 Token: {result1['usage']}")
    print(f"  📝 回复:\n{result1['content'][:500]}")

    # --- 测试 2: 进球检测 ---
    print(f"\n{'─' * 60}")
    print("📋 测试 2: 进球事件检测")
    print("   Prompt: '这段篮球视频中是否有进球得分发生？如有，发生在第几秒？'")
    print(f"{'─' * 60}")

    prompt2 = (
        "这是一段篮球比赛的视频。请仔细分析视频内容，找出所有投篮得分（进球）的瞬间。"
        "对于每个进球，请指出它发生在视频的第几秒（从视频开头算起）。"
        "请用以下JSON格式返回结果（不要包含其他文字）："
        '{"goals": [{"time_sec": 5, "description": "蓝衣30号上篮"}, ...], "total_goals": N}'
    )
    result2 = test_video_understanding(client, data_url, prompt2, max_tokens=8192)
    print(f"  ⏱️ 耗时: {result2['elapsed_s']}s")
    print(f"  🪙 Token: {result2['usage']}")
    print(f"  📝 回复:\n{result2['content']}")

    # 保存结果到 JSON 文件
    import re
    output_dir = Path(__file__).parent
    # 保存原始回复
    raw_file = output_dir / "mimo_response_raw.txt"
    raw_file.write_text(result2["content"], encoding="utf-8")
    print(f"\n  📄 原始回复已保存: {raw_file}")

    # 尝试解析 JSON 并保存结构化数据
    content = result2["content"]
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            goals_data = json.loads(json_match.group())
            goals_file = output_dir / "detected_goals.json"
            goals_file.write_text(json.dumps(goals_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  📄 进球数据已保存: {goals_file} ({goals_data.get('total_goals', 0)} 个进球)")
        except json.JSONDecodeError:
            print(f"  ⚠️ JSON 解析失败，请检查原始回复")

    # --- 汇总 ---
    print(f"\n{'=' * 60}")
    print("📊 验证汇总")
    print(f"{'=' * 60}")
    total_tokens = (result1.get("usage", {}).get("total_tokens", 0) +
                    result2.get("usage", {}).get("total_tokens", 0))
    total_time = result1["elapsed_s"] + result2["elapsed_s"]
    print(f"  总 Token 消耗: {total_tokens:,}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"\n✅ 验证结论:")
    print(f"  问题1 (视频输入) : {'通过' if result1['content'] else '失败'}")
    print(f"  问题2 (进球识别) : 请根据回复内容人工判断")
    print(f"  问题3 (时间定位) : 请根据回复内容人工判断")


if __name__ == "__main__":
    main()

"""
Basketball Highlight Reel Generator
篮球半场3V3进球集锦自动剪辑器

两阶段 Mimo v2.5 分析:
  Stage 1: 全视频低帧率扫描 → 检测所有进球时间戳
  Stage 2: 每个进球高帧率短片段 → 精细识别球员、球衣、2分/3分

用法: python highlight.py video.mp4 -o highlights.mp4
"""
import os
import re
import json
import base64
import subprocess
import time
import argparse
from pathlib import Path
from openai import OpenAI

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5"

# ── Stage 1: 进球检测 ────────────────────────────────────
GOAL_DETECTION_PROMPT = (
    "这是一段篮球比赛的视频。请仔细分析视频内容，找出所有投篮得分（进球）的瞬间。"
    "对于每个进球，请指出它发生在视频的第几秒（从视频开头算起）。"
    "请用以下JSON格式返回结果（不要包含其他文字）："
    '{"goals": [{"time_sec": 5, "description": "球员上篮"}, ...], "total_goals": N}'
)

# ── Stage 2: 球员精细识别 ────────────────────────────────
SCORER_ANALYSIS_PROMPT = (
    "这是一段篮球进球瞬间的短视频片段。请仔细观察投篮的球员，回答以下问题：\n"
    "1. 进球球员穿着什么颜色的球衣？（如：红色、白色、蓝色、黑色等）\n"
    "2. 球衣上是否有可见的号码？如果有，是多少号？\n"
    "3. 这个进球是几分球？（观察投篮位置——篮下/中距离=2分，三分线外=3分，罚球=1分）\n"
    "4. 投篮动作类型是什么？（上篮、跳投、三分远投、抛投、扣篮等）\n"
    "5. 简要描述进球球员的特征（如：红色球衣、身高较高、左手投篮等）\n\n"
    "请用以下JSON格式返回结果（不要包含其他文字）：\n"
    '{"jersey_color": "红色", "jersey_number": "23", "shot_type": "2分跳投",'
    '"action": "跳投", "scorer_desc": "红衣23号球员，身高较高"}'
)


def _call_mimo_video(client: OpenAI, data_url: str, prompt: str,
                     fps: float = 2, max_tokens: int = 4096) -> dict:
    """通用 Mimo 视频调用，返回 (content, elapsed_s, usage)"""
    start = time.time()
    response = client.chat.completions.create(
        model=MIMO_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": data_url},
                 "fps": fps, "media_resolution": "default"},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    usage = {}
    if hasattr(response, "usage") and response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return {"content": content, "elapsed_s": round(elapsed, 1), "usage": usage}


def encode_video_base64(path: Path) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return f"data:video/mp4;base64,{base64.b64encode(data).decode('utf-8')}"


def extract_clip(input_video: Path, start: float, duration: float,
                 output_path: Path) -> bool:
    """用 ffmpeg 截取视频片段"""
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start), "-i", str(input_video),
        "-t", str(duration), "-c", "copy",
        "-avoid_negative_ts", "make_zero", str(output_path),
    ], capture_output=True)
    return output_path.exists() and output_path.stat().st_size > 0


def _parse_json(content: str) -> dict:
    """从 Mimo 回复中提取 JSON，容错处理"""
    # 去掉 markdown 代码块标记
    cleaned = re.sub(r'```(?:json)?\s*', '', content)
    cleaned = cleaned.strip()
    # 找到最外层 JSON 对象
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if not json_match:
        raise ValueError(f"未找到 JSON:\n{content[:300]}")
    raw = json_match.group()
    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 尝试修复常见 JSON 错误：尾部多余逗号
    fixed = re.sub(r',\s*([}\]])', r'\1', raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    raise ValueError(f"JSON 解析失败:\n{raw[:300]}")


# ═══════════════════════════════════════════════════════════
# Stage 1: 进球时间戳检测
# ═══════════════════════════════════════════════════════════

def detect_goals(video_path: Path, api_key: str,
                 fps: float = 2, max_tokens: int = 8192) -> dict:
    """全视频低帧率扫描，返回所有进球时间戳"""
    client = OpenAI(api_key=api_key, base_url=MIMO_BASE_URL)

    print(f"  编码视频 ({video_path.stat().st_size / 1024 / 1024:.0f}MB)...")
    data_url = encode_video_base64(video_path)

    print(f"  发送到 Mimo v2.5 (fps={fps})...")
    result = _call_mimo_video(client, data_url, GOAL_DETECTION_PROMPT,
                              fps=fps, max_tokens=max_tokens)
    goals_data = _parse_json(result["content"])
    goals_data["_stage1_meta"] = result
    return goals_data


# ═══════════════════════════════════════════════════════════
# Stage 2: 逐球精细识别
# ═══════════════════════════════════════════════════════════

def refine_goals(video_path: Path, goals: list[dict], api_key: str,
                 refine_fps: float = 6, clip_window: float = 4.0,
                 cache_dir: Path = None) -> list[dict]:
    """对每个进球截取高帧率短片段，精细识别球员和投篮类型"""
    client = OpenAI(api_key=api_key, base_url=MIMO_BASE_URL)

    if cache_dir is None:
        cache_dir = video_path.parent / f"{video_path.stem}_refine_clips"
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    enriched = []

    for i, goal in enumerate(goals):
        t = goal["time_sec"]
        clip_start = max(0, t - 1.5)  # 进球前1.5秒
        clip_dur = min(clip_window, t + 2.5) if t < 2.5 else clip_window
        clip_path = cache_dir / f"refine_{i:03d}.mp4"

        print(f"  [{i+1}/{len(goals)}] @ {t}s ", end="", flush=True)

        if not extract_clip(video_path, clip_start, clip_dur, clip_path):
            print("⚠️ 裁剪失败")
            enriched.append(goal)
            continue

        data_url = encode_video_base64(clip_path)
        result = {}
        try:
            result = _call_mimo_video(client, data_url, SCORER_ANALYSIS_PROMPT,
                                      fps=refine_fps, max_tokens=1024)
            scorer = _parse_json(result["content"])
        except Exception as e:
            raw = result.get("content", "N/A")[:120]
            print(f"⚠️ {e} | raw: {raw}")
            enriched.append(goal)
            continue

        tokens = result["usage"].get("total_tokens", 0)
        total_tokens += tokens

        # 合并原始进球数据和精细分析结果
        enriched_goal = {**goal, **scorer}
        enriched.append(enriched_goal)

        jersey = scorer.get("jersey_color", "?")
        number = scorer.get("jersey_number", "?")
        shot = scorer.get("shot_type", "?")
        print(f"{jersey}{number}号 {shot} ({tokens}t)")

        # 清理临时片段
        clip_path.unlink(missing_ok=True)

    # 清理缓存目录
    try:
        cache_dir.rmdir()
    except OSError:
        pass

    return enriched


# ═══════════════════════════════════════════════════════════
# Phase C: CV 球员跟踪
# ═══════════════════════════════════════════════════════════

def track_players(video_path: Path, goals: list[dict],
                  track_window: float = 3.0) -> list[dict]:
    """CV 球员跟踪：YOLOv8 + ByteTrack。
    用 Mimo 的颜色描述匹配 CV 跟踪的球员，交叉验证投篮者身份。
    """
    from player_tracker import PlayerTracker

    print("  加载 YOLOv8 模型...")
    tracker = PlayerTracker()

    enriched = []
    for i, goal in enumerate(goals):
        t = goal["time_sec"]
        print(f"  [{i+1}/{len(goals)}] @ {t}s ", end="", flush=True)

        try:
            result = tracker.process_goal_clip(
                video_path, t, window=track_window
            )
        except Exception as e:
            print(f"⚠️ CV 跟踪失败: {e}")
            enriched.append(goal)
            continue

        # 交叉验证：Mimo 说什么颜色 → CV 找该颜色的球员
        mimo_color = goal.get("jersey_color", "")
        cv_players = result["all_players"]
        cv_colors = set(p["jersey_color"] for p in cv_players
                       if p["jersey_color"] != "unknown" and p["frames_tracked"] > 5)

        # 在 Mimo 声称的颜色中找最活跃的球员
        matched = [p for p in cv_players
                   if p["jersey_color"] == mimo_color and p["frames_tracked"] > 5]
        if matched:
            best = max(matched, key=lambda p: p["frames_tracked"])
            cv_scorer_color = best["jersey_color"]
            cv_scorer_id = best["track_id"]
            match_status = "✅" if mimo_color == cv_scorer_color else "⚠️"
        else:
            # 无匹配 → 用位置启发式兜底
            cv_scorer_color = result["scorer_jersey_color"]
            cv_scorer_id = result.get("scorer_track_id")
            match_status = "❓"

        enriched_goal = {
            **goal,
            "cv_scorer_color": cv_scorer_color,
            "cv_scorer_track_id": cv_scorer_id,
            "cv_all_players": cv_players,
            "cv_all_colors": sorted(cv_colors),
            "cv_match": match_status,
        }
        enriched.append(enriched_goal)

        print(f"Mimo={mimo_color} {match_status} CV={cv_scorer_color}"
              f" (场上颜色: {sorted(cv_colors)})")

    return enriched


# ═══════════════════════════════════════════════════════════
# 裁剪 + 拼接
# ═══════════════════════════════════════════════════════════

def _merge_ranges(goals: list[dict], before: float, after: float,
                  gap: float = 1.0) -> list[tuple[float, float, list[dict]]]:
    """合并重叠或相邻的进球时间窗口"""
    ranges = [(max(0, g["time_sec"] - before), g["time_sec"] + after, [g])
              for g in goals]
    ranges.sort(key=lambda r: r[0])
    merged = []
    cur_start, cur_end, cur_goals = ranges[0]
    for start, end, gs in ranges[1:]:
        if start < cur_end:
            # 时间真正重叠才合并
            cur_end = max(cur_end, end)
            cur_goals.extend(gs)
        elif gap > 0 and start - cur_end <= gap:
            # 用户指定了合并间距
            cur_end = max(cur_end, end)
            cur_goals.extend(gs)
        else:
            merged.append((cur_start, cur_end, cur_goals))
            cur_start, cur_end, cur_goals = start, end, gs
    merged.append((cur_start, cur_end, cur_goals))
    return merged


def clip_goals(input_video: Path, goals: list[dict], output_dir: Path,
               before: float = 5.0, after: float = 3.0,
               merge_gap: float = -1.0) -> list[Path]:
    """裁剪进球片段，自动合并重叠窗口"""
    ranges = _merge_ranges(goals, before, after, gap=merge_gap)

    if len(ranges) < len(goals):
        print(f"  🔗 重叠合并: {len(goals)} 个进球 → {len(ranges)} 个片段")

    clips = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (start, end, seg_goals) in enumerate(ranges):
        clip_path = output_dir / f"clip_{i:03d}.mp4"
        if extract_clip(input_video, start, end - start, clip_path):
            clips.append(clip_path)
            descs = " | ".join(
                g.get("scorer_desc", g.get("description", "?"))
                for g in seg_goals
            )
            print(f"  [{i+1}/{len(ranges)}] {start:.0f}s-{end:.0f}s "
                  f"({len(seg_goals)}球) {descs[:80]}")
        else:
            print(f"  [{i+1}/{len(ranges)}] ⚠️ 裁剪失败 @ {start:.0f}s-{end:.0f}s")

    return clips


def concat_clips(clips: list[Path], output_path: Path):
    """拼接所有片段为最终集锦"""
    concat_file = output_path.parent / "clips.txt"
    with open(concat_file, "w") as f:
        for clip in clips:
            f.write(f"file '{clip.absolute()}'\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), "-c", "copy", str(output_path),
    ], capture_output=True)
    concat_file.unlink()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="篮球进球集锦自动剪辑器")
    parser.add_argument("video", type=Path, help="输入视频文件")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="输出文件 (默认: <input>_highlights.mp4)")
    parser.add_argument("--before", type=float, default=5.0,
                        help="进球前截取秒数 (默认: 5)")
    parser.add_argument("--after", type=float, default=3.0,
                        help="进球后截取秒数 (默认: 3)")
    parser.add_argument("--fps", type=float, default=3,
                        help="Stage1 抽帧率 (默认: 3，越高越不容易漏球)")
    parser.add_argument("--refine-fps", type=float, default=6,
                        help="Stage2 精细分析抽帧率 (默认: 6)")
    parser.add_argument("--no-refine", action="store_true",
                        help="跳过 Stage2 精细球员识别")
    parser.add_argument("--track", action="store_true",
                        help="启用 CV 球员跟踪 (YOLOv8 + ByteTrack)")
    parser.add_argument("--track-window", type=float, default=3.0,
                        help="CV 跟踪分析窗口秒数 (默认: 3)")
    parser.add_argument("--no-concat", action="store_true",
                        help="只裁剪不拼接")
    parser.add_argument("--merge", type=float, default=-1.0,
                        help="合并间距小于N秒的相邻片段 (默认: -1 不合并)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Stage1 最大输出 token")
    args = parser.parse_args()

    api_key = os.environ.get("MIMO_API_KEY", "")
    if not api_key:
        print("❌ 请设置环境变量 MIMO_API_KEY")
        return
    if not args.video.exists():
        print(f"❌ 找不到视频文件: {args.video}")
        return
    if args.output is None:
        args.output = args.video.parent / f"{args.video.stem}_highlights.mp4"

    print("=" * 60)
    print("🏀 篮球进球集锦自动剪辑器")
    print("=" * 60)
    print(f"  输入: {args.video}")
    print(f"  输出: {args.output}")

    # ── Stage 1: 检测进球 ──
    print(f"\n🔍 Stage 1: 全视频进球检测 (fps={args.fps})...")
    try:
        data = detect_goals(args.video, api_key,
                            fps=args.fps, max_tokens=args.max_tokens)
    except Exception as e:
        print(f"❌ 进球检测失败: {e}")
        return

    goals = data.get("goals", [])
    s1_meta = data.get("_stage1_meta", {})

    # 获取视频实际时长，过滤越界进球
    import subprocess as _sp
    _dur = _sp.run(["ffprobe", "-v", "error", "-show_entries",
                     "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                     str(args.video)], capture_output=True, text=True)
    video_duration = float(_dur.stdout.strip()) if _dur.stdout.strip() else 0
    if video_duration > 0:
        valid = [g for g in goals if g["time_sec"] <= video_duration]
        if len(valid) < len(goals):
            print(f"  ⚠️ 过滤 {len(goals) - len(valid)} 个越界进球 (视频时长 {video_duration:.0f}s)")
        goals = valid

    print(f"  ✅ 检测到 {len(goals)} 个进球 "
          f"({s1_meta['usage'].get('total_tokens', 0):,}t, {s1_meta['elapsed_s']}s)")

    if not goals:
        print("  ℹ️ 未检测到进球，退出。")
        return

    # ── Stage 2: 精细识别 ──
    if not args.no_refine:
        print(f"\n🔬 Stage 2: 逐球精细识别 (fps={args.refine_fps})...")
        goals = refine_goals(args.video, goals, api_key,
                             refine_fps=args.refine_fps)
        print(f"  ✅ 精细识别完成")

    # ── Phase C: CV 球员跟踪 ──
    if args.track:
        print(f"\n👁️  Phase C: CV 球员跟踪 (YOLOv8 + ByteTrack)...")
        goals = track_players(args.video, goals,
                              track_window=args.track_window)
        print(f"  ✅ CV 跟踪完成")

    # ── 保存结果 ──
    result_file = args.output.parent / f"{args.video.stem}_goals.json"
    # 构建干净的输出（去掉内部元数据）
    output_data = {"goals": goals, "total_goals": len(goals)}
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  📄 进球数据: {result_file}")

    # ── 裁剪 + 拼接 ──
    print(f"\n✂️  裁剪进球片段 (±{args.before:.0f}s / +{args.after:.0f}s)...")
    clips_dir = args.output.parent / f"{args.video.stem}_clips"
    clips = clip_goals(args.video, goals, clips_dir,
                       before=args.before, after=args.after,
                       merge_gap=args.merge)
    print(f"  ✅ 成功裁剪 {len(clips)} 个片段")

    if not clips:
        print("❌ 没有成功裁剪的片段，退出。")
        return

    if not args.no_concat:
        print(f"\n🔗 拼接集锦...")
        concat_clips(clips, args.output)
        size_mb = args.output.stat().st_size / 1024 / 1024
        print(f"  ✅ 集锦已生成: {args.output} ({size_mb:.1f}MB)")

    print(f"\n{'=' * 60}")
    print("✅ 完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

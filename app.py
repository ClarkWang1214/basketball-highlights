"""
Streamlit Web UI for Basketball Highlights
启动: streamlit run app.py
"""
import os
import json
import time
import tempfile
from pathlib import Path

import streamlit as st

# 页面配置
st.set_page_config(
    page_title="Basketball Highlights",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀 Basketball Highlights")
st.caption("上传半场 3V3 比赛视频，自动生成进球集锦")

# ── 侧边栏：参数 ──────────────────────────
with st.sidebar:
    st.header("设置")
    api_key = st.text_input(
        "Mimo API Key",
        value=os.environ.get("MIMO_API_KEY", ""),
        type="password",
        help="从 https://platform.xiaomimimo.com 获取",
    )
    fps = st.slider("扫描帧率", 1, 5, 3,
                    help="越高越不容易漏球，token 消耗也越多")
    before = st.slider("进球前 (秒)", 2, 10, 5)
    after = st.slider("进球后 (秒)", 1, 5, 3)
    use_track = st.checkbox("CV 球员跟踪", value=True,
                            help="YOLO + ByteTrack 交叉验证球衣颜色")
    st.divider()
    st.caption("处理时长取决于视频长度，一般 2-5 分钟。")

# ── 主区域 ────────────────────────────────
video_file = st.file_uploader(
    "上传比赛视频",
    type=["mp4", "mov", "avi"],
    help="三脚架固定机位录制的半场 3V3 比赛",
)

if video_file:
    # 显示上传的视频
    col1, col2 = st.columns(2)
    with col1:
        st.video(video_file)
    with col2:
        st.metric("文件大小", f"{video_file.size / 1024 / 1024:.1f} MB")
        st.metric("文件名", video_file.name)

    if st.button("🚀 开始分析", type="primary", use_container_width=True):
        if not api_key:
            st.error("请先在左侧输入 Mimo API Key")
        else:
            # 保存上传的视频到临时文件
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as tmp:
                tmp.write(video_file.read())
                video_path = Path(tmp.name)

            output_dir = Path(tempfile.mkdtemp())
            output_path = output_dir / "highlights.mp4"

            try:
                # ── Stage 1: 检测进球 ──
                progress = st.status("🔍 检测进球中...", expanded=True)
                with progress:
                    st.write("Stage 1: Mimo v2.5 全视频扫描...")
                    from highlight import detect_goals, refine_goals, track_players

                    data = detect_goals(video_path, api_key, fps=fps)
                    goals = data.get("goals", [])
                    # 过滤越界进球
                    import subprocess
                    dur = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries",
                         "format=duration", "-of",
                         "default=noprint_wrappers=1:nokey=1",
                         str(video_path)],
                        capture_output=True, text=True,
                    )
                    vdur = float(dur.stdout.strip()) if dur.stdout.strip() else 0
                    valid = [g for g in goals if g["time_sec"] <= vdur]
                    st.write(f"检测到 {len(goals)} 个进球（有效 {len(valid)} 个）")
                    goals = valid

                if not goals:
                    st.warning("未检测到进球。试试降低扫描帧率或换一段视频。")
                else:
                    # ── Stage 2: 精细识别 ──
                    progress.update(label="🔬 精细识别中...")
                    goals = refine_goals(video_path, goals, api_key)
                    st.write(f"Stage 2: 完成 {len(goals)} 个进球精细识别")

                    # ── Stage 3: CV 跟踪 ──
                    if use_track:
                        progress.update(label="👁️ 球员跟踪中...")
                        goals = track_players(video_path, goals)
                        st.write(f"Stage 3: CV 跟踪完成")
                    progress.update(
                        label="✅ 分析完成",
                        state="complete",
                    )

                    # ── 裁剪拼接 ──
                    progress2 = st.status("✂️ 生成集锦中...", expanded=True)
                    from highlight import clip_goals, concat_clips

                    clips_dir = output_dir / "clips"
                    clips = clip_goals(
                        video_path, goals, clips_dir,
                        before=before, after=after,
                    )
                    concat_clips(clips, output_path)
                    progress2.update(
                        label="✅ 集锦生成完成",
                        state="complete",
                    )

                    # ── 展示结果 ──
                    st.success(f"🎉 完成！检测到 {len(goals)} 个进球")

                    # 进球列表
                    with st.expander(f"📋 {len(goals)} 个进球详情"):
                        for g in goals:
                            t = g["time_sec"]
                            m, s = int(t // 60), int(t % 60)
                            color = g.get("jersey_color", "?")
                            shot = g.get("shot_type", "?")
                            cv = g.get("cv_match", "")
                            st.write(f"`{m:02d}:{s:02d}` {color} {shot} {cv}")

                    # 集锦视频
                    st.subheader("📺 进球集锦")
                    st.video(str(output_path))

                    # 下载按钮
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "⬇️ 下载集锦 (MP4)",
                            data=f,
                            file_name="highlights.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )

                    # 进球数据下载
                    goals_json = json.dumps(
                        {"goals": goals, "total_goals": len(goals)},
                        ensure_ascii=False, indent=2,
                    )
                    st.download_button(
                        "📄 下载进球数据 (JSON)",
                        data=goals_json,
                        file_name="goals.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"处理失败: {e}")
                import traceback
                with st.expander("错误详情"):
                    st.code(traceback.format_exc())
            finally:
                # 清理临时文件
                import shutil
                video_path.unlink(missing_ok=True)
                shutil.rmtree(output_dir, ignore_errors=True)

"""Caption synchronized UAV frames and render a tiled video."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg
import ollama
from PIL import Image, ImageDraw, ImageFont

from .pipeline import dump, read_frames

PROMPT = (
    "Describe this UAV camera image in one concise English sentence, at most 24 words. "
    "Focus on clearly visible roads, buildings, vehicles, people, vegetation, and their "
    "relative positions. Mention only visible facts. Do not infer hidden objects, "
    "intentions, or motion from a still image. Return only the caption without a heading."
)


def render(args):
    data, output = args.data.resolve(), args.output.resolve()
    capture = json.loads((data / "capture.json").read_text())
    frames = read_frames(data)
    fps = capture["fps"]
    stride = round(args.interval * fps)
    if stride < 1 or not math.isclose(stride / fps, args.interval):
        raise ValueError("Caption interval must contain a whole number of frames")
    grouped = {}
    for row in frames:
        grouped.setdefault(row["uav_id"], []).append(row)
    uavs = sorted(grouped)
    count = len(grouped[uavs[0]])
    timeline = [r["time_s"] for r in grouped[uavs[0]]]
    if any([r["time_s"] for r in grouped[u]] != timeline for u in uavs):
        raise ValueError("UAV frame timelines must match")
    if args.columns < 1 or args.tile_width < 64 or args.tile_width % 2:
        raise ValueError("Use positive columns and an even tile width of at least 64")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    client = ollama.Client(host=args.ollama, timeout=180)
    model_info = next(m.model_dump(mode="json") for m in client.list().models if m.model == args.model)
    prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()
    captions = {}
    for uav in uavs:
        captions[uav] = []
        folder = work / f"uav_{uav:02d}"
        folder.mkdir(exist_ok=True)
        for index in range(0, count, stride):
            row = grouped[uav][index]
            image_path = data / row["image"]
            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            checkpoint = folder / f"{index:06d}.json"
            if checkpoint.exists():
                item = json.loads(checkpoint.read_text())
                if (item["image_sha256"], item["prompt_sha256"], item["model_digest"]) != (
                        image_hash, prompt_hash, model_info["digest"]):
                    raise ValueError(f"Caption checkpoint does not match input: {checkpoint}")
            else:
                started = time.monotonic()
                response = client.chat(model=args.model, stream=False, keep_alive="10m",
                    messages=[{"role": "user", "content": PROMPT, "images": [str(image_path)]}],
                    options={"temperature": 0, "num_ctx": 4096, "num_predict": 128, "seed": 42})
                text = response.message.content.strip()
                if not text or response.done_reason == "length":
                    raise RuntimeError(f"Empty or truncated caption for {row['id']}")
                item = {"uav_id": uav, "index": index, "start_s": index / fps,
                        "end_s": min(index + stride, count) / fps, "caption": text,
                        "image": row["image"], "image_sha256": image_hash,
                        "prompt_sha256": prompt_hash, "model_digest": model_info["digest"],
                        "response": response.model_dump(mode="json"), "wall_seconds": time.monotonic() - started}
                dump(checkpoint, item)
            captions[uav].append(item)
            print(f"UAV {uav:02d} {index / fps:04.1f}s: {item['caption']}", flush=True)
    tile_w = args.tile_width
    tile_h = round(tile_w * capture["height"] / capture["width"] / 2) * 2
    width, height = tile_w * args.columns, tile_h * math.ceil(len(uavs) / args.columns)
    font_size = round(tile_w * 30 / 1024)
    font = ImageFont.truetype(str(args.font), font_size)
    inset, spacing = round(tile_w * 22 / 1024), round(tile_w * 9 / 1024)
    overlays = {}
    for uav, items in captions.items():
        overlays[uav] = []
        for item in items:
            lines, line = [], ""
            for word in f"UAV {uav}: {item['caption']}".split():
                candidate = (line + " " + word).strip()
                if line and font.getlength(candidate) > tile_w - 2 * inset:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
            layer = Image.new("RGBA", (tile_w, tile_h))
            draw = ImageDraw.Draw(layer)
            text = "\n".join(lines)
            box = draw.multiline_textbbox((inset, inset), text, font=font, spacing=spacing)
            band = box[3] + inset
            if band > tile_h * .4:
                raise ValueError(f"Caption exceeds tile overlay for UAV {uav}")
            draw.rectangle((0, 0, tile_w, band), fill=(0, 0, 0, 180))
            draw.multiline_text((inset, inset), text, font=font, fill="white", spacing=spacing)
            overlays[uav].append(layer)
    manifest = {"completed": False, "model": model_info, "prompt": PROMPT, "uavs": uavs,
                "fps": fps, "frames": count, "duration_s": count / fps,
                "caption_interval_s": args.interval, "caption_count": sum(map(len, captions.values())),
                "layout": {"columns": args.columns, "rows": math.ceil(len(uavs) / args.columns),
                           "width": width, "height": height, "font_size_px": font_size},
                "capture_sha256": hashlib.sha256((data / "capture.json").read_bytes()).hexdigest(),
                "output": str(output)}
    dump(work / "video.json", manifest)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for index in range(count):
            mosaic = Image.new("RGB", (width, height))
            for position, uav in enumerate(uavs):
                with Image.open(data / grouped[uav][index]["image"]) as original:
                    tile = original.resize((tile_w, tile_h), Image.Resampling.LANCZOS).convert("RGBA")
                tile = Image.alpha_composite(tile, overlays[uav][index // stride]).convert("RGB")
                mosaic.paste(tile, ((position % args.columns) * tile_w, (position // args.columns) * tile_h))
            encoder.stdin.write(mosaic.tobytes())
            if index in {0, count // 2, count - 1}:
                mosaic.save(work / f"preview_{index:06d}.jpg", quality=95)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
    manifest.update(completed=True, video_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    dump(work / "video.json", manifest)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-vl:8b-instruct")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
    render(parser.parse_args())


if __name__ == "__main__":
    main()

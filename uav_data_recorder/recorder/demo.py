"""Make a captioned presentation video from a completed panorama dataset.

Acquisition lives in panorama.py; this module only performs offline FPV
projection, VLM captioning, layout and video rendering.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

from .panorama import ffmpeg_path, fpv_maps

PROMPT = ('Describe this rectified UAV camera image in one concise English sentence, '
          'at most 24 words. State only clearly visible objects and their spatial '
          'arrangement. Do not infer motion, intentions, identities, or hidden '
          'objects. Return only the caption, without a heading.')
COLORS = {1: '#ffcf47', 2: '#45a8ff', 3: '#ff606b'}


def wrap_caption(text, font, width=600):
    lines, current = [], ''
    for word in text.split():
        candidate = (current+' '+word).strip()
        if current and font.getlength(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines+[current] if current else lines


def caption_overlay(uid, caption, font_path, speed):
    layer = Image.new('RGBA', (640, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, 0, 639, 79), fill=(12, 18, 26, 255))
    draw.rectangle((0, 400, 639, 439), fill=(12, 18, 26, 255))
    draw.rectangle((0, 0, 639, 3), fill=COLORS[uid])
    draw.text((20, 7), f'UAV{uid}', font=ImageFont.truetype(font_path, 31), fill=COLORS[uid])
    draw.text((562, 12), f'{speed:g}×', font=ImageFont.truetype(font_path, 25), fill='white')
    small = ImageFont.truetype(font_path, 20)
    draw.text((20, 49), 'PANORAMIC CAMERA', font=small, fill='#d4dce8')
    draw.text((20, 405), 'RECTIFIED FPV', font=small, fill='#d4dce8')
    font = ImageFont.truetype(font_path, 26)
    lines = wrap_caption(caption, font)
    if len(lines) > 5 or any(font.getlength(line) > 600 for line in lines):
        raise ValueError('Caption exceeds layout; refusing to truncate original model text')
    text = '\n'.join(lines)
    bounds = draw.multiline_textbbox((20, 456), text, font=font, spacing=7)
    draw.rectangle((0, 440, 639, bounds[3]+15), fill=(0, 0, 0, 180))
    draw.multiline_text((20, 456), text, font=font, spacing=7, fill='white')
    draw.line((639, 0, 639, 1079), fill='#6b727c', width=2)
    rgba = np.asarray(layer)
    return rgba[:, :, :3][:, :, ::-1].copy(), rgba[:, :, 3:4].astype(np.float32)/255


def read_panorama(directory, uid, index, size):
    path = directory/'panorama_frames'/f'uav{uid}'/f'{index:06d}.png'
    image = cv2.imread(str(path))
    if image is None or image.shape != (size[1], size[0], 3):
        raise ValueError(f'Missing/invalid panorama frame: {path}')
    return image


def prepare_caption_frames(args, capture):
    """Derive sampled FPV frames for the demo without changing captured data."""
    paths = {}
    interval = round(capture['caption_interval_source_s']*capture['fps'])
    native = capture.get('schema_version', 1) == 2
    if native:
        maps = fpv_maps(capture['panorama_size'][0], pitch=capture['fpv_pitch'])
        folder = args.work_dir/'caption_frames'
        folder.mkdir(parents=True, exist_ok=True)
    for uid in (1, 2, 3):
        for index in range(0, capture['frames_per_uav'], interval):
            if native:
                panorama = read_panorama(args.capture, uid, index, capture['panorama_size'])
                fpv = cv2.remap(panorama, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                path = folder/f'uav{uid}_{index:06d}.png'
                if path.exists():
                    previous = cv2.imread(str(path))
                    if previous is None or not np.array_equal(previous, fpv):
                        raise ValueError(f'Incompatible demo frame cache: {path}; use a new output directory')
                elif not cv2.imwrite(str(path), fpv):
                    raise OSError(f'Could not save demo frame: {path}')
            else:
                path = args.capture/'caption_frames'/f'uav{uid}_{index:06d}.jpg'
                if not path.is_file():
                    raise FileNotFoundError(path)
            paths[uid, index] = path
    return paths


def caption_capture(args, capture):
    response = requests.get(args.ollama+'/api/tags', timeout=15)
    response.raise_for_status()
    model = next((m for m in response.json()['models'] if m['name'] == args.model), None)
    if model is None:
        raise ValueError(f'Model not installed in Ollama: {args.model}')
    cache = args.work_dir/'responses'
    cache.mkdir(exist_ok=True)
    captions = {}
    interval = round(capture['caption_interval_source_s']*capture['fps'])
    for uid in (1, 2, 3):
        captions[uid] = []
        for index in range(0, capture['frames_per_uav'], interval):
            image = args.caption_paths[uid, index]
            content = image.read_bytes()
            identity = dict(image_sha256=hashlib.sha256(content).hexdigest(),
                            prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                            model_digest=model['digest'])
            path = cache/f'uav{uid}_{index:06d}.json'
            if path.exists():
                record = json.loads(path.read_text())
                if any(record[k] != v for k, v in identity.items()):
                    raise ValueError(f'Incompatible caption cache {path}')
            else:
                response = requests.post(args.ollama+'/api/chat', json=dict(
                    model=args.model, stream=False, keep_alive='10m',
                    messages=[dict(role='user', content=PROMPT, images=[base64.b64encode(content).decode()])],
                    options=dict(temperature=0, seed=42, num_ctx=4096, num_predict=128)), timeout=240)
                response.raise_for_status()
                result = response.json()
                caption = result['message']['content'].strip()
                if not caption or result.get('done_reason') == 'length':
                    raise RuntimeError('Empty/truncated VLM output')
                record = dict(identity, caption=caption, response=result, uav=uid, index=index,
                              image=str(image.resolve()), source_time_s=index/capture['fps'])
                path.write_text(json.dumps(record, indent=2)+'\n')
            captions[uid].append(dict(record, output_start_s=index/capture['fps']/args.playback_speed))
            print(f'UAV{uid} source {index/capture["fps"]:g}s: {record["caption"]}', flush=True)
    return captions, model


def render_demo(args):
    capture = json.loads((args.capture/'capture.json').read_text())
    if not capture.get('completed') or capture['uavs'] != 3:
        raise ValueError('A completed three-UAV panorama capture is required')
    version = capture.get('schema_version', 1)
    if version not in (1, 2):
        raise ValueError(f'Unsupported capture schema: {version}')
    native = version == 2
    if native:
        capture['caption_interval_source_s'] = args.interval if args.interval is not None else 8
        capture['fpv_pitch'] = args.fpv_pitch if args.fpv_pitch is not None else -20
    elif args.interval is not None or args.fpv_pitch is not None:
        raise ValueError('Legacy captures have baked-in FPV/interval; omit --interval and --fpv-pitch')
    interval = capture['caption_interval_source_s']*capture['fps']
    if not np.isfinite(interval) or interval < 1 or not np.isclose(interval, round(interval)):
        raise ValueError('Caption interval must be a positive whole number of capture frames')
    if not np.isfinite(capture['fpv_pitch']):
        raise ValueError('FPV pitch must be finite')
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
        raise ValueError('Positive finite playback speed required')
    if not args.font.is_file():
        raise FileNotFoundError('Provide --font /path/to/DejaVuSans-Bold.ttf')
    target = args.output or args.capture/'demo'/'three_uav_carla_panorama_fpv_captioned_1080p.mp4'
    if target.exists():
        raise FileExistsError(target)
    args.work_dir = target.parent
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.caption_paths = prepare_caption_frames(args, capture)
    captions, model = caption_capture(args, capture)
    layers = {uid: [caption_overlay(uid, c['caption'], str(args.font), args.playback_speed) for c in records]
              for uid, records in captions.items()}
    caps = [] if native else [cv2.VideoCapture(str(args.capture/f'uav{uid}_panorama_fpv_raw.mp4')) for uid in (1, 2, 3)]
    for cap in caps:
        if (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != capture['frames_per_uav']
                or abs(cap.get(cv2.CAP_PROP_FPS)-capture['fps']) > 1e-6):
            for opened in caps:
                opened.release()
            raise ValueError('Raw video frame count/FPS mismatch')
    maps = fpv_maps(capture['panorama_size'][0], pitch=capture['fpv_pitch']) if native else None
    # Preserve every sampled simulation frame; change its playback timestamp.
    fps = capture['fps']*args.playback_speed
    frames = capture['frames_per_uav']
    interval = round(capture['caption_interval_source_s']*capture['fps'])
    ffmpeg = ffmpeg_path(args.ffmpeg)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '1920x1080',
        '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '6',
        '-preset', 'fast', '-crf', '20', '-maxrate', '20M', '-bufsize', '40M',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], stdin=subprocess.PIPE)
    try:
        for index in range(frames):
            columns = []
            for uid in (1, 2, 3):
                if native:
                    panorama = read_panorama(args.capture, uid, index, capture['panorama_size'])
                    fpv = cv2.remap(panorama, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                else:
                    ok, row = caps[uid-1].read()
                    if not ok or row.shape[:2] != (720, 2160):
                        raise RuntimeError(f'Missing/invalid raw frame {uid}/{index}')
                    panorama, fpv = row[:, :1440], row[:, 1440:]
                column = np.zeros((1080, 640, 3), np.uint8)
                column[80:400] = cv2.resize(panorama, (640, 320), interpolation=cv2.INTER_AREA)
                column[440:] = cv2.resize(fpv, (640, 640), interpolation=cv2.INTER_AREA)
                rgb, alpha = layers[uid][index//interval]
                columns.append((column*(1-alpha)+rgb*alpha).astype(np.uint8))
            frame = np.hstack(columns)
            encoder.stdin.write(frame.tobytes())
            if index in (0, frames//2, frames-1):
                cv2.imwrite(str(target.parent/f'demo_preview_{index:04d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if index % 60 == 0:
                print(f'Render {index}/{frames} frames', flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('Composite encoding failed')
    finally:
        for cap in caps:
            cap.release()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
    check = cv2.VideoCapture(str(target))
    count = 0
    actual_fps = check.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = check.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            raise ValueError('Invalid final dimensions')
        count += 1
    check.release()
    if count != frames or abs(fps-actual_fps) > 1e-6:
        raise ValueError('Final frame count/FPS mismatch')
    manifest = dict(output=str(target.resolve()), capture=str(args.capture.resolve()),
        resolution=[1920, 1080], frames=count, fps=fps, duration_s=count/fps,
        playback_speed=args.playback_speed, source_duration_s=capture['duration_s'],
        capture_schema_version=version, fpv_pitch=capture['fpv_pitch'],
        caption_interval_output_s=capture['caption_interval_source_s']/args.playback_speed,
        captions=captions, model=model, prompt=PROMPT,
        caption_semantics='Offline VLM inference, text held until next sampled frame; original output, not live inference latency',
        layout='UAV1/2/3 left-to-right; panorama above rectified FPV; no crop/stretch; top black-alpha180/white-bold captions',
        validation='All output frames decoded; dimensions/FPS/count checked', audio='omitted')
    target.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'PASS: {count} frames, {fps:g} FPS, {count/fps:g}s: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--playback-speed', type=float, default=2)
    parser.add_argument('--interval', type=float, help='Caption interval in source seconds (new datasets: 8)')
    parser.add_argument('--fpv-pitch', type=float, help='FPV pitch in degrees (new datasets: -20)')
    parser.add_argument('--ollama', default='http://127.0.0.1:11434')
    parser.add_argument('--model', default='qwen3-vl:8b-instruct')
    parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    parser.add_argument('--ffmpeg')
    render_demo(parser.parse_args())


if __name__ == '__main__':
    main()

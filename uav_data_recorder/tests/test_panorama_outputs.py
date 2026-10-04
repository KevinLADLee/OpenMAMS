"""Real PNG/FFmpeg IO with deterministic fake CARLA sensor delivery; no server/VLM."""
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from recorder import demo, panorama, routes


@pytest.fixture
def captured(tmp_path, monkeypatch):
    carla = pytest.importorskip('carla')
    actors = []
    settings = NS(synchronous_mode=False, fixed_delta_seconds=None)
    weather = object()

    class Camera:
        def __init__(self, transform):
            self.id = len(actors)
            self.transform = transform
            self.stopped = self.destroyed = False

        def listen(self, callback):
            self.callback = callback

        def stop(self):
            self.stopped = True

        def destroy(self):
            self.destroyed = True

    class World:
        ticks = 0

        def get_map(self):
            return NS(name='Town05')

        def get_settings(self):
            return NS(**vars(settings))

        def apply_settings(self, value):
            self.final_settings = value

        def get_weather(self):
            return weather

        def set_weather(self, value):
            self.final_weather = value

        def get_blueprint_library(self):
            return NS(find=lambda _: NS(set_attribute=lambda *_: None))

        def spawn_actor(self, bp, transform):
            camera = Camera(transform)
            actors.append(camera)
            return camera

        def tick(self, timeout):
            self.ticks += 1
            for camera in actors:
                pixels = np.full((32, 32, 4), (camera.id*11+self.ticks)%256, np.uint8)
                camera.callback(NS(frame=self.ticks, timestamp=self.ticks/10,
                                   transform=camera.transform, raw_data=pixels.tobytes()))
            return self.ticks

    world = World()

    def apply_batch(batch, tick):
        for index, transform in batch:
            actors[index].transform = transform
        return [NS(error=None) for _ in batch]

    client = NS(set_timeout=lambda _: None, get_world=lambda: world,
                get_client_version=lambda: 'fake', get_server_version=lambda: 'fake',
                apply_batch_sync=apply_batch)
    monkeypatch.setattr(carla, 'Client', lambda *_: client)
    monkeypatch.setattr(carla, 'command', NS(ApplyTransform=lambda uid, pose: (uid, pose)))
    route = tmp_path/'route.json'
    route.write_text(json.dumps(dict(map='Town05', closed=True,
                                    route_waypoints=[[0, 0, 0], [30, 0, 0], [30, 30, 0]])))
    output = tmp_path/'capture'
    args = NS(uavs=3, fps=10, seconds=.5, speed=4.5, altitude=12, face_size=32,
              pano_width=128, ffmpeg=None, output=output, host='localhost', port=2000,
              map='Town05', route=route)
    panorama.capture_panorama(args)
    assert all(c.stopped and c.destroyed for c in actors)
    assert world.final_settings.synchronous_mode is False
    assert world.final_weather is weather
    return output


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return fps, frames


def test_capture_pure_panorama_and_every_png(captured):
    manifest = json.loads((captured/'capture.json').read_text())
    assert manifest['completed'] and manifest['schema_version'] == 2
    assert manifest['panorama_size'] == [128, 64]
    records = [json.loads(line) for line in (captured/'frames.jsonl').read_text().splitlines()]
    assert len(records) == 15
    assert not (captured/'caption_frames').exists()
    for uid in (1, 2, 3):
        pngs = sorted((captured/'panorama_frames'/f'uav{uid}').glob('*.png'))
        assert [p.name for p in pngs] == [f'{i:06d}.png' for i in range(5)]
        fps, decoded = video_info(captured/f'uav{uid}_panorama.mp4')
        assert fps == 10 and len(decoded) == 5
        assert all(frame.shape == (64, 128, 3) for frame in decoded)
        assert not (captured/f'uav{uid}_panorama_fpv_raw.mp4').exists()
    for record in records:
        assert (captured/record['panorama_image']).is_file()
        assert record['face_frames'] == [record['world_frame']]*6
        assert record['time_s'] == pytest.approx(record['index']/10)
        assert record['timestamp'] == pytest.approx(manifest['start_timestamp']+record['time_s'])


def test_png_is_lossless_and_refuses_overwrite(tmp_path):
    frame = np.random.default_rng(7).integers(0, 256, (32, 64, 3), dtype=np.uint8)
    writer = NS(stdin=io.BytesIO())
    relative = panorama.save_panorama_frame(tmp_path, 1, 0, frame, writer)
    assert np.array_equal(cv2.imread(str(tmp_path/relative)), frame)
    assert writer.stdin.getvalue() == frame.tobytes()
    with pytest.raises(FileExistsError):
        panorama.save_panorama_frame(tmp_path, 1, 0, frame, writer)


def test_failed_png_write_is_not_encoded(tmp_path, monkeypatch):
    writer = NS(stdin=io.BytesIO())
    monkeypatch.setattr(cv2, 'imwrite', lambda *_: False)
    with pytest.raises(OSError):
        panorama.save_panorama_frame(tmp_path, 1, 0, np.zeros((32, 64, 3), np.uint8), writer)
    assert not writer.stdin.getvalue()


@pytest.mark.parametrize('legacy', [False, True])
def test_demo_native_and_legacy(captured, monkeypatch, legacy):
    manifest_path = captured/'capture.json'
    capture = json.loads(manifest_path.read_text())
    if legacy:
        capture.pop('schema_version')
        capture.update(fpv_pitch=-20, caption_interval_source_s=.2)
        manifest_path.write_text(json.dumps(capture))
        (captured/'caption_frames').mkdir()
        for uid in (1, 2, 3):
            writer = panorama.video_writer(panorama.ffmpeg_path(None),
                                            captured/f'uav{uid}_panorama_fpv_raw.mp4', 2160, 720, 10)
            for i in range(5):
                frame = np.full((720, 2160, 3), 40+uid*30+i, np.uint8)
                writer.stdin.write(frame.tobytes())
                if i%2 == 0:
                    cv2.imwrite(str(captured/'caption_frames'/f'uav{uid}_{i:06d}.jpg'), frame[:, 1440:])
            writer.stdin.close()
            assert writer.wait() == 0
    original_manifest = manifest_path.read_bytes()
    # Tests exercise real caption cache/encoding, never invoke a live VLM.
    posts = []
    def response(data):
        return NS(raise_for_status=lambda: None, json=lambda: data)
    monkeypatch.setattr(demo.requests, 'get', lambda *a, **k: response(
        {'models': [{'name': 'test-model', 'digest': 'fixture-only'}]}))
    def post(*a, **kwargs):
        posts.append(kwargs)
        return response({'message': {'content': 'Synthetic test scene.'}, 'done_reason': 'stop'})
    monkeypatch.setattr(demo.requests, 'post', post)
    args = NS(capture=captured, output=None, playback_speed=2,
              interval=None if legacy else .2, fpv_pitch=None, ffmpeg=None,
              font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
              ollama='http://test.invalid', model='test-model')
    demo.render_demo(args)
    target = captured/'demo'/'three_uav_carla_panorama_fpv_captioned_1080p.mp4'
    fps, frames = video_info(target)
    assert fps == 20 and len(frames) == 5 and frames[0].shape == (1080, 1920, 3)
    assert len(posts) == 9
    assert manifest_path.read_bytes() == original_manifest
    result = json.loads(target.with_suffix('.json').read_text())
    assert result['duration_s'] == .25
    assert result['capture_schema_version'] == (1 if legacy else 2)


def test_missing_panorama_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='Missing/invalid'):
        demo.read_panorama(tmp_path, 1, 0, [128, 64])


def test_legacy_module_alias():
    from recorder import panorama_demo
    assert panorama_demo.main is demo.main

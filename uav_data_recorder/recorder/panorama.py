"""Capture pure spherical panorama videos and every lossless panorama frame.

Three kinematic virtual camera rigs, not a UAV flight-dynamics simulation.
All six faces share an optical center; every emitted rig shares a world frame.
"""
import argparse
from contextlib import suppress
import json
import math
from pathlib import Path
import queue
import subprocess

import numpy as np

# name, CARLA yaw/pitch, forward/right/up in rig-local CARLA XYZ (Z up).
FACES = (
    ('front', 0, 0, (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ('right', 90, 0, (0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    ('back', 180, 0, (-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    ('left', -90, 0, (0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ('up', 0, 90, (0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    ('down', 0, -90, (0, 0, -1), (0, 1, 0), (1, 0, 0)),
)


def spherical_maps(face_size, width):
    """Pixel-center equirectangular rays -> six 90-degree pinhole images."""
    if face_size < 32 or width < 64 or width % 2:
        raise ValueError('Even panorama width >=64 and face size >=32 required')
    height = width//2
    lon = ((np.arange(width)+.5)/width-.5)*2*np.pi
    lat = (.5-(np.arange(height)+.5)/height)*np.pi
    x = np.cos(lat)[:, None]*np.cos(lon)[None, :]
    y = np.cos(lat)[:, None]*np.sin(lon)[None, :]
    z = np.broadcast_to(np.sin(lat)[:, None], x.shape)
    rays = np.stack([x, y, z], axis=-1)
    forwards = np.array([face[3] for face in FACES])
    owner = np.argmax(rays@forwards.T, axis=-1)
    maps = []
    for i, (_, _, _, forward, right, up) in enumerate(FACES):
        mask = owner == i
        selected = rays[mask]
        depth = selected@np.array(forward)
        u = face_size/2*(selected@np.array(right)/depth+1)-.5
        v = face_size/2*(1-selected@np.array(up)/depth)-.5
        maps.append((mask, u.astype(np.float32)[None, :], v.astype(np.float32)[None, :]))
    return maps


def stitch_faces(images, maps, width):
    import cv2
    panorama = np.empty((width//2, width, 3), dtype=np.uint8)
    if len(images) != 6:
        raise ValueError('Six synchronized cube faces required')
    for image, (mask, u, v) in zip(images, maps):
        # remap has a signed-short size limit; chunk flattened coordinates.
        flat_u, flat_v = u.ravel(), v.ravel()
        samples = np.empty((len(flat_u), 3), np.uint8)
        for start in range(0, len(flat_u), 16000):
            end = start+16000
            samples[start:end] = cv2.remap(image, flat_u[start:end][None], flat_v[start:end][None],
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
        panorama[mask] = samples
    return panorama


def fpv_maps(width, size=720, pitch=-20, fov=90):
    if not -89 <= pitch <= 89 or not 1 < fov < 179:
        raise ValueError('Invalid FPV pitch/FOV')
    grid = (2*(np.arange(size)+.5)/size-1)*np.tan(np.deg2rad(fov/2))
    r, down = np.meshgrid(grid, grid)
    p = np.deg2rad(pitch)
    forward, up = np.array([np.cos(p), 0, np.sin(p)]), np.array([-np.sin(p), 0, np.cos(p)])
    rays = forward+r[..., None]*np.array([0, 1, 0])-down[..., None]*up
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    lon = np.arctan2(rays[..., 1], rays[..., 0])
    lat = np.arcsin(rays[..., 2])
    return ((lon/(2*np.pi)+.5)*width-.5).astype(np.float32), ((.5-lat/np.pi)*(width//2)-.5).astype(np.float32)


def get_frame(q, expected, timeout=60):
    while True:
        image = q.get(timeout=timeout)
        if image.frame == expected:
            return image
        if image.frame > expected:
            raise RuntimeError(f'Sensor ahead of world: {image.frame} > {expected}')


def rotation_basis_error(actual, expected):
    """Euler angles are non-unique at cube poles; compare orthonormal bases."""
    return max(np.linalg.norm(np.array([a.x, a.y, a.z])-np.array([b.x, b.y, b.z]))
               for a, b in ((actual.get_forward_vector(), expected.get_forward_vector()),
                            (actual.get_right_vector(), expected.get_right_vector()),
                            (actual.get_up_vector(), expected.get_up_vector())))


def ffmpeg_path(value):
    if value:
        return value
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def video_writer(ffmpeg, target, width, height, fps):
    return subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{width}x{height}',
        '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2',
        '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)],
        stdin=subprocess.PIPE)


def save_panorama_frame(output, uid, index, panorama, writer):
    """Archive the full-resolution stitched pixels before video compression."""
    import cv2
    relative = Path('panorama_frames')/f'uav{uid}'/f'{index:06d}.png'
    target = Path(output)/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    if not cv2.imwrite(str(target), panorama, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f'Could not save panorama frame: {target}')
    writer.stdin.write(panorama.tobytes())
    return relative.as_posix()


def capture_panorama(args):
    import carla
    import cv2
    from .routes import load_routes

    if (args.uavs < 1 or args.fps < 10 or args.seconds <= 0 or args.speed <= 0
            or args.altitude <= 0):
        raise ValueError('Positive parameters and FPS >=10 required')
    if not all(math.isfinite(v) for v in (args.fps, args.seconds, args.speed, args.altitude)):
        raise ValueError('Parameters must be finite')
    if args.pano_width % 4:
        raise ValueError('Panorama width must be divisible by 4 for YUV420 video')
    frames = round(args.seconds*args.fps)
    if not math.isclose(frames/args.fps, args.seconds):
        raise ValueError('Duration must contain a whole number of frames')
    maps = spherical_maps(args.face_size, args.pano_width)
    ffmpeg = ffmpeg_path(args.ffmpeg)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Use a fresh output directory: {output}')
    client = carla.Client(args.host, args.port)
    client.set_timeout(120)
    world = client.get_world()
    if world.get_map().name.rsplit('/', 1)[-1] != args.map:
        world = client.load_world(args.map)
    routes, route_info = load_routes(args.route, args.map, args.uavs, frames, args.fps, args.speed)
    output.mkdir(parents=True)
    (output/'panorama_frames').mkdir()
    (output/'previews').mkdir()
    (output/'route.json').write_text(json.dumps(route_info['source'], indent=2)+'\n')
    old_settings, old_weather = world.get_settings(), world.get_weather()
    cameras, queues, writers = [], [], []
    yaws = [route[0].transform.rotation.yaw for route in routes]
    start_timestamp = None
    manifest = dict(schema_version=2, completed=False, map=args.map, uavs=args.uavs, frames_per_uav=frames,
                    fps=args.fps, duration_s=args.seconds, face_size=args.face_size,
                    panorama_size=[args.pano_width, args.pano_width//2],
                    panorama_video_pattern='uav{uav}_panorama.mp4',
                    panorama_frame_pattern='panorama_frames/uav{uav}/{index:06d}.png',
                    image_encoding='Lossless PNG of every stitched panorama; no FPV or overlays',
                    video_encoding='H.264 CRF19 YUV420; lossy playback copy at original capture FPS',
                    altitude_above_road_m=args.altitude, speed_mps=args.speed,
                    source='CARLA synchronous sensor.camera.rgb, six co-located 90deg faces per UAV',
                    camera_model='Kinematic virtual UAV rigs; no flight dynamics or visible UAV bodies',
                    synchronization='All six faces of all UAVs have identical world_frame and timestamp',
                    stitching='Direction-vector cubemap to 2:1 equirectangular',
                    photometry='Shared manual exposure: ISO100, shutter200, f/4, compensation0, gamma2.2; no motion blur or lens distortion',
                    route_length_m=route_info['length_m'], route_offsets_m=route_info['offsets_m'],
                    carla_client_version=client.get_client_version(), carla_server_version=client.get_server_version())
    (output/'capture.json').write_text(json.dumps(manifest, indent=2)+'\n')
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1/args.fps
        settings.substepping = True
        settings.max_substep_delta_time = .01
        settings.max_substeps = 10
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        for uav in range(args.uavs):
            p = routes[uav][0].transform.location
            for name, yaw, pitch, *_ in FACES:
                bp = world.get_blueprint_library().find('sensor.camera.rgb')
                attributes = dict(image_size_x=args.face_size, image_size_y=args.face_size, fov=90,
                                  sensor_tick=0, role_name=f'panorama_uav{uav+1}_{name}',
                                  enable_postprocess_effects='true', exposure_mode='manual',
                                  shutter_speed=200, iso=100, fstop=4, exposure_compensation=0,
                                  motion_blur_intensity=0, lens_k=0, lens_kcube=0,
                                  lens_circle_multiplier=0, gamma=2.2)
                for key, value in attributes.items():
                    bp.set_attribute(key, str(value))
                camera = world.spawn_actor(bp, carla.Transform(
                    carla.Location(x=p.x, y=p.y, z=p.z+args.altitude),
                    carla.Rotation(yaw=yaws[uav]+yaw, pitch=pitch)))
                cameras.append(camera)
                q = queue.Queue()
                queues.append(q)
                camera.listen(q.put)
            writers.append(video_writer(ffmpeg, output/f'uav{uav+1}_panorama.mp4',
                                        args.pano_width, args.pano_width//2, args.fps))
        # Drain every sensor during warmup so no stale frame enters the recording.
        for _ in range(12):
            frame = world.tick(120)
            for q in queues:
                get_frame(q, frame)
        with (output/'frames.jsonl').open('w') as stream:
            for index in range(frames):
                transforms = []
                for uav in range(args.uavs):
                    pose = routes[uav][index].transform
                    yaws[uav] += .15*((pose.rotation.yaw-yaws[uav]+180)%360-180)
                    for _, yaw, pitch, *_ in FACES:
                        transforms.append(carla.Transform(
                            carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z+args.altitude),
                            carla.Rotation(yaw=yaws[uav]+yaw, pitch=pitch)))
                result = client.apply_batch_sync([carla.command.ApplyTransform(c.id, t)
                                                 for c, t in zip(cameras, transforms)], False)
                if any(r.error for r in result):
                    raise RuntimeError([r.error for r in result if r.error])
                frame = world.tick(120)
                images = [get_frame(q, frame) for q in queues]
                if len({image.timestamp for image in images}) != 1:
                    raise RuntimeError('Cross-face/UAV timestamp mismatch')
                if start_timestamp is None:
                    start_timestamp = images[0].timestamp
                if abs(images[0].timestamp-start_timestamp-index/args.fps) > .002:
                    raise RuntimeError('Simulation timestamp drift')
                for image, expected in zip(images, transforms):
                    if image.transform.location.distance(expected.location) > .001:
                        raise RuntimeError('Actual sensor location differs from requested rig center')
                    if rotation_basis_error(image.transform.rotation, expected.rotation) > .001:
                        raise RuntimeError('Actual camera orientation mismatch')
                for uav in range(args.uavs):
                    faces = [np.frombuffer(im.raw_data, np.uint8).reshape(args.face_size, args.face_size, 4)[:, :, :3]
                             for im in images[uav*6:(uav+1)*6]]
                    panorama = stitch_faces(faces, maps, args.pano_width)
                    panorama_image = save_panorama_frame(output, uav+1, index, panorama, writers[uav])
                    if index in (0, frames//2, frames-1):
                        cv2.imwrite(str(output/'previews'/f'uav{uav+1}_{index:06d}_panorama.jpg'), panorama)
                        if index == 0:
                            for face, pixels in zip(FACES, faces):
                                cv2.imwrite(str(output/'previews'/f'uav{uav+1}_cube_{face[0]}.jpg'), pixels)
                    p = images[uav*6].transform.location
                    stream.write(json.dumps(dict(uav=uav+1, index=index, time_s=index/args.fps,
                        timestamp=images[uav*6].timestamp, world_frame=frame, face_frames=[i.frame for i in images[uav*6:(uav+1)*6]],
                        x=p.x, y=p.y, z=p.z, yaw=yaws[uav], panorama_image=panorama_image,
                        panorama_video=f'uav{uav+1}_panorama.mp4'))+'\n')
                if index % args.fps == 0:
                    print(f'Captured {index+1}/{frames} synchronized frames per UAV', flush=True)
        for writer in writers:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError('Raw video encoder failed')
        manifest.update(completed=True, start_timestamp=start_timestamp)
        (output/'capture.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'Capture complete: {output}', flush=True)
    finally:
        for writer in writers:
            if writer.poll() is None:
                with suppress(Exception):
                    writer.stdin.close()
                writer.terminate()
                with suppress(Exception):
                    writer.wait(timeout=10)
        client.set_timeout(5)
        for camera in cameras:
            with suppress(Exception):
                camera.stop()
            with suppress(Exception):
                camera.destroy()
        with suppress(Exception):
            world.apply_settings(old_settings)
            world.set_weather(old_weather)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--map', default='Town05')
    parser.add_argument('--route', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--uavs', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=40)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--speed', type=float, default=4.5)
    parser.add_argument('--altitude', type=float, default=12)
    parser.add_argument('--face-size', type=int, default=768)
    parser.add_argument('--pano-width', type=int, default=2048)
    parser.add_argument('--ffmpeg')
    capture_panorama(parser.parse_args())


if __name__ == '__main__':
    main()

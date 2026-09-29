"""Standalone CARLA RGB capture using road waypoints and virtual UAV cameras."""
from contextlib import suppress
import json
import math
from pathlib import Path
import queue
import random

import numpy as np
from PIL import Image

TARGETS = [
    ("green car", "vehicle.tesla.model3", "0,255,0"),
    ("blue car", "vehicle.tesla.model3", "0,0,255"),
    ("white car", "vehicle.tesla.model3", "255,255,255"),
    ("purple car", "vehicle.tesla.model3", "128,0,128"),
    ("black car", "vehicle.tesla.model3", "0,0,0"),
    ("fire truck", "vehicle.carlamotors.firetruck", None),
    ("motorcycle", "vehicle.yamaha.yzf", None),
    ("bus", "vehicle.mitsubishi.fusorosa", None),
    ("taxi", "vehicle.ford.crown", None),
    ("traffic cone", "static.prop.trafficcone01", None),
]


def record(args):
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError('Install the optional collector: pip install -e ".[carla]"') from exc
    for key in ("seconds", "fps", "speed", "altitude", "fov", "pitch"):
        if not math.isfinite(getattr(args, key)):
            raise ValueError(f"{key} must be finite")
    count = len(args.spawns) if args.spawns else args.uavs
    if count < 1 or args.fps < 1 or args.seconds <= 0 or args.speed <= 0 or args.altitude <= 0:
        raise ValueError("UAV count, FPS, duration, speed and altitude must be positive; FPS >= 1")
    if args.width < 32 or args.height < 32 or not 1 < args.fov < 179 or not -90 <= args.pitch < 0:
        raise ValueError("Invalid camera size, FOV or downward pitch")
    frames = round(args.seconds * args.fps)
    if frames < 1 or not math.isclose(frames / args.fps, args.seconds):
        raise ValueError("Duration must contain a whole number of frames")
    if args.objects and (frames < 30 or args.speed * args.seconds < 60):
        raise ValueError("Ten-object scenes need at least 30 frames and 60m of inspected road per UAV")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    client = carla.Client(args.host, args.port)
    client.set_timeout(60)
    world = client.get_world()
    if world.get_map().name.rsplit("/", 1)[-1] != args.map:
        world = client.load_world(args.map)
    road = world.get_map()
    rng = random.Random(args.seed)
    route_info = None
    if getattr(args, "route", None):
        from .routes import load_routes
        routes, route_info = load_routes(args.route, args.map, count, frames, args.fps, args.speed)
        indices = []
    else:
        spawns, rng = road.get_spawn_points(), random.Random(args.seed)
        if args.spawns:
            indices = args.spawns
            if len(set(indices)) != count or any(i < 0 or i >= len(spawns) for i in indices):
                raise ValueError("Spawn indices must be distinct and valid for this map")
        else:
            candidates = list(range(len(spawns)))
            rng.shuffle(candidates)
            indices = []
            for i in candidates:
                if all(spawns[i].location.distance(spawns[j].location) >= 80 for j in indices):
                    indices.append(i)
                if len(indices) == count:
                    break
            if len(indices) != count:
                raise ValueError("Cannot find enough separated spawn points; pass --spawns explicitly")
        routes = []
        for start in indices:
            route = [road.get_waypoint(spawns[start].location)]
            for _ in range(frames - 1):
                choices = sorted(route[-1].next(args.speed / args.fps), key=lambda w: (w.road_id, w.lane_id, w.s))
                if not choices:
                    raise ValueError(f"Road ends before requested duration; choose another spawn than {start}")
                route.append(rng.choice(choices))
            routes.append(route)
    output.mkdir(parents=True)
    if route_info is not None:
        (output / "route.json").write_text(json.dumps(route_info["source"], indent=2) + "\n")
    old_settings, old_weather = world.get_settings(), world.get_weather()
    owned, cameras, queues, truth = [], [], [], []
    yaws = [r[0].transform.rotation.yaw for r in routes]
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        # For sparse collection, tick physics more frequently than the camera sample rate.
        sub_ticks = max(1, math.ceil(10 / args.fps))
        dt = 1 / (args.fps * sub_ticks)
        settings.fixed_delta_seconds = dt
        settings.substepping = True
        settings.max_substep_delta_time = .01
        settings.max_substeps = 10
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        sensor_tick = 0  # Sample on fixed world ticks to avoid sparse sensor timer drift.
        if args.objects:
            assignment = [i % count for i in range(10)]
            rng.shuffle(assignment)
            for number, ((name, blueprint, color), uav) in enumerate(zip(TARGETS, assignment), 1):
                for _ in range(300):
                    waypoint = routes[uav][rng.randrange(max(1, frames // 8), frames - 1)]
                    p = waypoint.transform.location
                    if any(math.hypot(p.x - o["x"], p.y - o["y"]) < 14 for o in truth):
                        continue
                    bp = world.get_blueprint_library().find(blueprint)
                    if color:
                        bp.set_attribute("color", color)
                    pose = carla.Transform(carla.Location(x=p.x, y=p.y, z=p.z + 1),
                                           carla.Rotation(yaw=waypoint.transform.rotation.yaw))
                    actor = world.try_spawn_actor(bp, pose)
                    if actor is None:
                        continue
                    owned.append(actor)
                    actor.set_simulate_physics(False)
                    box = actor.bounding_box
                    pose.location.z = p.z + .03 - (box.location.z - box.extent.z)
                    result = client.apply_batch_sync([carla.command.ApplyTransform(actor.id, pose)], False)
                    if result[0].error:
                        raise RuntimeError(result[0].error)
                    truth.append(dict(id=number, label=name, blueprint=blueprint, color=color,
                                      x=p.x, y=p.y, z=pose.location.z, assigned_uav=uav + 1))
                    break
                else:
                    raise RuntimeError(f"No free road position for {name}; increase duration or change seed")
            (output / "objects.json").write_text(json.dumps(truth, indent=2) + "\n")
        for index, route in enumerate(routes):
            bp = world.get_blueprint_library().find("sensor.camera.rgb")
            for key, value in dict(image_size_x=args.width, image_size_y=args.height,
                                    fov=args.fov, sensor_tick=sensor_tick, role_name=f"uav_{index+1}").items():
                bp.set_attribute(key, str(value))
            p = route[0].transform.location
            camera = world.spawn_actor(bp, carla.Transform(carla.Location(x=p.x, y=p.y, z=p.z + args.altitude),
                                      carla.Rotation(pitch=args.pitch, yaw=yaws[index])))
            owned.append(camera)
            cameras.append(camera)
            q = queue.Queue()
            queues.append(q)
            camera.listen(q.put)
            (output / "images" / f"uav_{index+1:02d}").mkdir(parents=True)
        start_timestamp = None
        with (output / "frames.jsonl").open("w") as stream:
            for frame_index in range(frames):
                for sub in range(sub_ticks):
                    commands, expected = [], []
                    for i, (camera, route) in enumerate(zip(cameras, routes)):
                        a, b = route[max(0, frame_index-1)].transform.location, route[frame_index].transform.location
                        fraction = (sub + 1) / sub_ticks
                        p = carla.Location(x=a.x+(b.x-a.x)*fraction, y=a.y+(b.y-a.y)*fraction,
                                           z=a.z+(b.z-a.z)*fraction+args.altitude)
                        target = route[frame_index].transform.rotation.yaw
                        yaws[i] += .5 * ((target - yaws[i] + 180) % 360 - 180)
                        commands.append(carla.command.ApplyTransform(camera.id,
                            carla.Transform(p, carla.Rotation(pitch=args.pitch, yaw=yaws[i]))))
                        expected.append(p)
                    result = client.apply_batch_sync(commands, False)
                    if any(r.error for r in result):
                        raise RuntimeError([r.error for r in result])
                    world_frame = world.tick(60)
                    if sub != sub_ticks - 1:
                        continue
                    images = []
                    for q in queues:
                        while True:
                            image = q.get(timeout=60)
                            if image.frame == world_frame:
                                break
                            if image.frame > world_frame:
                                raise RuntimeError("Camera frame ahead of world tick")
                        images.append(image)
                    if len({image.timestamp for image in images}) != 1:
                        raise RuntimeError("Camera timestamps are not synchronized")
                    if start_timestamp is None:
                        start_timestamp = images[0].timestamp
                    for i, image in enumerate(images):
                        p, rotation = image.transform.location, image.transform.rotation
                        if p.distance(expected[i]) > .001:
                            raise RuntimeError("Actual camera pose differs from commanded pose")
                        pixels = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
                        name = f"images/uav_{i+1:02d}/{frame_index:06d}.jpg"
                        Image.fromarray(pixels[:, :, [2, 1, 0]]).save(output / name, quality=90)
                        stream.write(json.dumps(dict(id=f"uav{i+1}_{frame_index:06d}", uav_id=i+1,
                            time_s=frame_index/args.fps, timestamp=image.timestamp, world_frame=world_frame,
                            x=p.x, y=p.y, z=p.z, yaw=rotation.yaw, pitch=rotation.pitch, image=name)) + "\n")
                if frame_index % 50 == 0:
                    print(f"Captured {frame_index+1}/{frames} frames per UAV", flush=True)
        metadata = dict(completed=True, map=road.name, uavs=count, frames_per_uav=frames, fps=args.fps,
                        duration_s=frames/args.fps, spawn_indices=indices, seed=args.seed, objects=len(truth),
                        width=args.width, height=args.height, fov=args.fov, pitch=args.pitch,
                        speed_mps=args.speed, altitude_above_road_m=args.altitude,
                        sensor_tick=sensor_tick, simulation_step_s=dt, capture_interval_s=1/args.fps,
                        start_timestamp=start_timestamp, camera_model="kinematic virtual RGB cameras")
        if route_info is not None:
            metadata.update(route_length_m=route_info["length_m"], route_offsets_m=route_info["offsets_m"])
        (output / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
    finally:
        client.set_timeout(3)
        for camera in cameras:
            with suppress(Exception):
                camera.stop()
        for actor in reversed(owned):
            with suppress(Exception):
                actor.destroy()
        with suppress(Exception):
            world.apply_settings(old_settings)
            world.set_weather(old_weather)

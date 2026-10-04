"""CPU-only projection and synchronization regression tests (no CARLA needed)."""
import queue
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from recorder.panorama import FACES, spherical_maps, stitch_faces, fpv_maps, get_frame


def test_cube_orientation_and_seams():
    size, width = 128, 512
    xy = 2*(np.arange(size)+.5)/size-1
    right, down = np.meshgrid(xy, xy)
    faces = []
    for _, _, _, forward, r, up in FACES:
        ray = np.array(forward)+right[..., None]*r-down[..., None]*up
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        faces.append(np.rint((ray+1)*127.5).astype(np.uint8))
    maps = spherical_maps(size, width)
    assert np.all(sum(m[0].astype(int) for m in maps) == 1)
    actual = stitch_faces(faces, maps, width).astype(float)
    lon = ((np.arange(width)+.5)/width-.5)*2*np.pi
    lat = (.5-(np.arange(width//2)+.5)/(width//2))*np.pi
    x = np.cos(lat)[:, None]*np.cos(lon)[None, :]
    y = np.cos(lat)[:, None]*np.sin(lon)[None, :]
    z = np.broadcast_to(np.sin(lat)[:, None], x.shape)
    expected = (np.stack([x, y, z], axis=-1)+1)*127.5
    assert np.abs(actual-expected).max() < 2
    assert np.abs(actual[:, 0]-actual[:, -1]).max() < 4


@pytest.mark.parametrize('pitch', [0, -20, 30])
def test_rectified_fpv_center(pitch):
    u, v = fpv_maps(2048, size=101, pitch=pitch)
    assert u[50, 50] == pytest.approx(1023.5, abs=.001)
    assert v[50, 50] == pytest.approx((.5-pitch/180)*1024-.5, abs=.001)
    assert u[50, 0] < u[50, -1]
    assert v[0, 50] < v[-1, 50]


def test_sync_discards_old_and_rejects_future():
    q = queue.Queue()
    for i in (1, 2, 3):
        q.put(SimpleNamespace(frame=i))
    assert get_frame(q, 2).frame == 2
    with pytest.raises(RuntimeError, match='ahead'):
        get_frame(q, 2)


def test_invalid_projection_sizes():
    with pytest.raises(ValueError):
        spherical_maps(16, 512)
    with pytest.raises(ValueError):
        spherical_maps(128, 513)


def test_carla_rotation_basis_if_installed():
    carla = pytest.importorskip('carla')
    for _, yaw, pitch, forward, right, up in FACES:
        rotation = carla.Rotation(yaw=yaw, pitch=pitch)
        for actual, expected in ((rotation.get_forward_vector(), forward),
                                 (rotation.get_right_vector(), right),
                                 (rotation.get_up_vector(), up)):
            assert np.allclose([actual.x, actual.y, actual.z], expected, atol=1e-6)


def test_equivalent_euler_poles_and_wrong_orientation():
    carla = pytest.importorskip('carla')
    from recorder.panorama import rotation_basis_error
    expected = carla.Rotation(pitch=90, yaw=135, roll=0)
    # Actual CARLA 0.9.16 sensor representation from the short capture probe.
    actual = carla.Rotation(pitch=90, yaw=-165.963730, roll=59.036194)
    assert rotation_basis_error(actual, expected) < .001
    assert rotation_basis_error(carla.Rotation(pitch=90, yaw=45), expected) > 1


def test_caption_style_preserves_all_words():
    from recorder.demo import caption_overlay, wrap_caption
    from PIL import ImageFont
    from pathlib import Path
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    if not font_path.exists():
        pytest.skip('Reference font unavailable')
    caption = 'A paved road curves between tall buildings, with trees and parked cars along the sidewalk.'
    font = ImageFont.truetype(str(font_path), 26)
    assert ' '.join(wrap_caption(caption, font)) == caption
    rgb, alpha = caption_overlay(1, caption, str(font_path), 2)
    assert rgb.shape == (1080, 640, 3)
    assert float(alpha[445, 30, 0]) == pytest.approx(180/255)

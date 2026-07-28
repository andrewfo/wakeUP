import numpy as np

from wakeUp import geo


def test_haversine_known_distance():
    # ~1 degree of latitude ≈ 111.19 km
    d = geo.haversine_m(0.0, 0.0, 1.0, 0.0)
    assert abs(d - 111_195) < 200


def test_destination_roundtrip():
    lat0, lon0 = 36.5, -74.0
    for bearing in (0, 45, 90, 180, 270, 359):
        lat2, lon2 = geo.destination_point(lat0, lon0, bearing, 5000.0)
        back = geo.haversine_m(lat0, lon0, lat2, lon2)
        assert abs(back - 5000.0) < 1.0


def test_bearing_cardinal():
    assert abs(geo.initial_bearing_deg(0, 0, 1, 0) - 0.0) < 1e-6      # north
    assert abs(geo.initial_bearing_deg(0, 0, 0, 1) - 90.0) < 1e-6     # east


def test_angular_diff_wraps():
    assert geo.angular_diff_deg(10, 350) == 20
    assert geo.angular_diff_deg(350, 10) == -20
    assert abs(geo.angular_diff_deg(0, 180)) == 180


def test_knot_conversion_roundtrip():
    v = np.array([0.0, 5.0, 12.5, 40.0])
    assert np.allclose(geo.knots_to_ms(geo.ms_to_knots(v)), v)

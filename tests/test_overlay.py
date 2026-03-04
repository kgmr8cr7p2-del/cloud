from overlay.overlay import OverlayWindow


def test_screen_to_client_uses_monitor_offset():
    ov = OverlayWindow()
    ov._monitor_rect = (1920, 0, 2560, 1440)

    assert ov._screen_to_client(1920, 0) == (0, 0)
    assert ov._screen_to_client(2000, 120) == (80, 120)


def test_screen_to_client_supports_negative_monitor_origin():
    ov = OverlayWindow()
    ov._monitor_rect = (-1280, 0, 1280, 1024)

    assert ov._screen_to_client(-1280, 0) == (0, 0)
    assert ov._screen_to_client(-1000, 300) == (280, 300)


def test_set_monitor_rect_updates_internal_rect_without_window():
    ov = OverlayWindow()
    ov.set_monitor_rect((100, 200, 1280, 720))

    assert ov._monitor_rect == (100, 200, 1280, 720)

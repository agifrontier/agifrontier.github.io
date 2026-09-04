import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "generate_image_metadata.py"
SPEC = importlib.util.spec_from_file_location("generate_image_metadata", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def webp(chunk: bytes, payload: bytes) -> bytes:
    body = b"WEBP" + chunk + len(payload).to_bytes(4, "little") + payload
    return b"RIFF" + len(body).to_bytes(4, "little") + body


class WebpDimensionsTest(unittest.TestCase):
    def assert_dimensions(self, data: bytes, expected: tuple[int, int]) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.webp"
            path.write_bytes(data)
            self.assertEqual(MODULE.intrinsic_dimensions(path), expected)

    def test_vp8x_dimensions(self) -> None:
        width, height = 1600, 900
        payload = b"\x00\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
        self.assert_dimensions(webp(b"VP8X", payload), (width, height))

    def test_vp8l_dimensions(self) -> None:
        width, height = 797, 498
        bits = (width - 1) | ((height - 1) << 14)
        self.assert_dimensions(webp(b"VP8L", b"\x2f" + bits.to_bytes(4, "little")), (width, height))

    def test_vp8_dimensions(self) -> None:
        width, height = 1200, 675
        payload = b"\x00\x00\x00\x9d\x01\x2a" + width.to_bytes(2, "little") + height.to_bytes(2, "little")
        self.assert_dimensions(webp(b"VP8 ", payload), (width, height))


if __name__ == "__main__":
    unittest.main()

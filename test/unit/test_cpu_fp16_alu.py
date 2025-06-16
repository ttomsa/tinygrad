import unittest
from tinygrad import Tensor, dtypes
from tinygrad.device import Device
from tinygrad.helpers import OSX

class TestFloat16Alu(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "CPU" and OSX, "")
  def test_cpu(self):
    a = Tensor([1,2,3,4], dtype=dtypes.float16)
    b = Tensor([1,2,3,4], dtype=dtypes.float16)
    c = (a + b).realize()

if __name__ == "__main__":
  unittest.main()
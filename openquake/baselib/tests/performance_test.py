import time
import unittest
import tempfile
import shutil
from openquake.baselib.performance import PerformanceMonitor


class MonitorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmpdir = tempfile.mkdtemp()
        cls.mon = PerformanceMonitor('test', tmpdir)

    def test_no_mem(self):
        mon = self.mon('test_no_mem')
        for i in range(3):
            with mon:
                time.sleep(0.1)
        self.assertGreater(mon.duration, 0.3)
        mon.flush()

    def test_mem(self):
        mon = self.mon('test_mem', measuremem=True)
        ls = []
        for i in range(3):
            with mon:
                ls.append(list(range(100000)))  # allocate some RAM
                time.sleep(0.1)
        self.assertGreaterEqual(mon.mem, 0)
        mon.flush()

    def test_children(self):
        mon = PerformanceMonitor('test', tempfile.mkdtemp())
        mon1 = mon('child1')
        mon2 = mon('child2')
        with mon1:
            time.sleep(0.1)
        with mon2:
            time.sleep(0.1)
        mon.flush()
        data = mon.collect_performance()
        total_time = data['time_sec'].sum()
        self.assertGreaterEqual(total_time, 0.2)
        shutil.rmtree(mon.monitor_dir)

    @classmethod
    def tearDownClass(cls):
        data = cls.mon.collect_performance()
        assert len(data) == 2
        assert data['time_sec'].sum() > 0
        assert data['memory_mb'].sum() >= 0
        shutil.rmtree(cls.mon.monitor_dir)

"""Regression tests for data loss, label semantics and file boundaries."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from util.loghub_romailika import load_loghub
from util.eda_romailika import window_table, analyze


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def load(self, name):
        with patch('util.loghub_romailika.fetch_loghub', return_value=self.root):
            return load_loghub(name, self.root)

    def test_multiline_and_file_boundaries(self):
        self.write('abnormal_label.txt', '### WordCount\nNormal:\n+ application_1_1\n')
        self.write('application_1_1/a.log', '2015-10-17 15:37:56,547 INFO [main] component: start\n\tat stack\n\n')
        self.write('application_1_1/b.log', 'orphan\n2015-10-17 15:37:57,000 ERROR [main] component: stop\n')
        b = self.load('Hadoop')
        self.assertEqual(b.quality['physical_lines'], 5)
        self.assertEqual(len(b.events), 3)
        self.assertEqual(int(b.events.physical_lines.sum()), 5)
        self.assertEqual(b.events.iloc[0].message, 'start\n\tat stack\n')
        self.assertFalse(b.events.iloc[1].parse_ok)
        self.assertEqual(b.events.iloc[2].context_label, 'Normal')

    def test_openstack_context_is_not_blanket_anomaly_label(self):
        vm = '544fd51c-4edc-4780-baae-ba1d80a0acfc'
        self.write('anomaly_labels.txt', vm)
        prefix = 'nova.log 2017-05-14 19:39:01.445 12 INFO nova.compute '
        self.write('openstack_abnormal.log', prefix + f'[instance: {vm}] failed\n' + prefix + 'ordinary event\n' + prefix.rstrip() + '\n')
        b = self.load('OpenStack')
        self.assertEqual(b.events.context_label.tolist(), ['injected VM', 'unlabelled in abnormal capture', 'unlabelled in abnormal capture'])
        self.assertEqual(b.events.iloc[2].message, '')
        self.assertFalse(b.events.timestamp.isna().any())
        self.assertNotIn('is_anomaly', b.events)

    def test_hpc_daemon_prefix_and_unlabelled_numeric_field(self):
        self.write('HPC.log', '1 node-1 unix.hw error 1145552216 1 failure\n2 daemon node-2 server subsys 1145552221 -1 warning\n3 node-D7 fs stopped 1135.5 1 broken time\n')
        b = self.load('HPC')
        self.assertEqual(b.events.host.tolist(), ['node-1', 'node-2', 'node-D7'])
        self.assertTrue(pd.isna(b.events.timestamp.iloc[2]))
        self.assertEqual(b.events.numeric_flag.iloc[1], '-1')
        self.assertEqual(str(b.events.timestamp.dt.tz), 'UTC')
        self.assertNotIn('is_anomaly', b.events)

    def test_android_synthetic_year_and_invalid_utf8_not_dropped(self):
        (self.root / 'Android.log').write_bytes(b'12-17 19:31:36.263  1795 1825 I Tag: hi\n\xffbroken\n')
        b = self.load('Android')
        self.assertEqual(b.events.timestamp.iloc[0].year, 2000)
        self.assertEqual(b.quality['invalid_utf8_lines'], 1)
        self.assertEqual(b.quality['orphan_lines'], 1)
        self.assertEqual(len(b.events), 2)

    def test_android_missing_tag_separator_retains_timestamp(self):
        self.write('Android.log', '12-17 19:31:36.263 1 2 I missing separator message\n')
        b = self.load('Android')
        self.assertFalse(b.events.timestamp.isna().any())
        self.assertEqual(b.events.component.iloc[0], 'unknown')
        self.assertEqual(b.quality['missing_tag_separator'], 1)

    def test_reservoir_reproducible_and_not_just_head(self):
        self.write('Android.log', ''.join(f'12-17 19:31:36.263 1 2 I Tag: {i}\n' for i in range(500)))
        a, b = self.load('Android'), self.load('Android')
        self.assertTrue(a.raw_random.equals(b.raw_random))
        self.assertEqual(len(a.raw_random), 100)
        self.assertGreater(a.raw_random.line_no.max(), 100)

    def test_windows_do_not_bridge_independent_files(self):
        df = pd.DataFrame({'source_file': ['a', 'a', 'b'],
                           'timestamp': pd.to_datetime(['2020-01-01 00:00', '2020-01-01 00:02', '2021-01-01 00:00'])})
        row = window_table(df).set_index('window').loc['1min']
        self.assertEqual(row.windows, 4)
        self.assertEqual(row['empty'], 1)
        self.assertEqual(row['max'], 1)

    def test_analysis_exports_consistent_counts(self):
        self.write('Android.log', '12-17 19:31:36.263 1 2 I Tag: number 100\n12-17 19:32:36.263 1 2 I Tag: number 200\n')
        bundle = self.load('Android')
        result = analyze(bundle, self.root / 'reports')
        self.assertEqual(result['summary']['logical_events'], 2)
        self.assertEqual(result['summary']['unique_templates'], 1)
        self.assertTrue((self.root / 'reports' / 'entities.csv').exists())


if __name__ == '__main__':
    unittest.main()

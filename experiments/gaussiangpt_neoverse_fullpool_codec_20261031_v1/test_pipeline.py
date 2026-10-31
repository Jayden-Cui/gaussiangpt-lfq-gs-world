"""CPU assertions for resource/coverage gates; no server calls or model execution."""
import unittest
import pipeline

class QueueTests(unittest.TestCase):
    def setUp(self):
        self.p={'resources':{'minimum_free_gib':55,'minimum_disk_free_gib':80}}

    def test_both_gpus_and_disk_must_pass_without_reduction(self):
        value=pipeline.check_resources('0, 70000\n1, 65000\n',90*2**30,self.p)
        self.assertEqual(set(value['free_mib']),{0,1})
        for output,space in [('0, 70000',90*2**30),('0, 70000\n1, 50000',90*2**30),
                             ('0, 70000\n1, 65000',79*2**30)]:
            with self.assertRaises(RuntimeError):pipeline.check_resources(output,space,self.p)

    def test_terminal_ledger_rejects_missing_duplicate_failed_or_timeout(self):
        good=[dict(name=n,exit_code=0,timed_out=False) for n in ('a','b')]
        self.assertTrue(pipeline.successful(good,['a','b']))
        cases=[good[:1],[good[0],good[0]],[good[0],dict(good[1],exit_code=1)],
               [good[0],dict(good[1],timed_out=True)]]
        for rows in cases:self.assertFalse(pipeline.successful(rows,['a','b']))

if __name__=='__main__':unittest.main()

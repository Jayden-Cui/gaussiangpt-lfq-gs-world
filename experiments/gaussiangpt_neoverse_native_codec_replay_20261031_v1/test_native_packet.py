"""CPU checks for the packet boundary and corruption rejection."""
import tempfile
import unittest
from pathlib import Path

import torch
from native_packet import load_packet, read_checked, save_new


class PacketBoundaryTest(unittest.TestCase):
    def sample(self):
        return dict(schema='native_codec_gs_packet_v1', clip='buildingblock_tower_17_176', frame=44,
                    checkpoint_sha256='a'*64, state_sha256='b'*64, source_cache_sha256='c'*64,
                    gs=torch.ones(3, 14), sh=torch.ones(3, 4, 3), cameras=torch.eye(4).repeat(2, 1, 1),
                    intrinsics=torch.ones(4), near=.1, far=100., pts=1., resolution_hw=[252,448],
                    normalization={'scale': .5}, decoded_from_actual_state=True)

    def test_round_trip_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'p.pt'
            meta = save_new(path, self.sample())
            raw = read_checked(path, meta['bytes'], meta['sha256'])
            self.assertEqual(load_packet(raw, 44, 'a'*64, 'b'*64)['gs'].shape, (3, 14))
            with self.assertRaises(ValueError): load_packet(raw, 45, 'a'*64, 'b'*64)
            with self.assertRaises(ValueError): load_packet(raw, 44, 'a'*64, 'd'*64)
            with self.assertRaises(ValueError): read_checked(path, meta['bytes'], '0'*64)

    def test_reject_nonfinite_and_reduced_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'p.pt'
            packet = self.sample()
            packet['gs'][0, 0] = float('nan')
            meta = save_new(path, packet)
            with self.assertRaises(ValueError): load_packet(read_checked(path, **{'size':meta['bytes'], 'digest':meta['sha256']}), 44, 'a'*64, 'b'*64)
            path = Path(directory)/'q.pt'
            packet = self.sample()
            packet['cameras'] = packet['cameras'][:1]
            meta = save_new(path, packet)
            with self.assertRaises(ValueError): load_packet(read_checked(path, meta['bytes'], meta['sha256']), 44, 'a'*64, 'b'*64)


if __name__ == '__main__': unittest.main()

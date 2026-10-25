"""Coordinator ownership tests; no CLI request or GPU work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cycle2_check as coordinator


class SerializedSlot(unittest.TestCase):
    def test_active_attention_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            slot=Path(d)/'slot.json'
            raw=json.dumps(dict(state='accepted',problem='muse_attention_decode',experiment='E5',job_id='existing'))
            slot.write_text(raw)
            with patch.object(coordinator,'GPU_SLOT',slot):
                with self.assertRaises(RuntimeError):
                    coordinator.gpu_slot('sending',dict(experiment='P038',submission_sha256='source'))
            self.assertEqual(slot.read_text(),raw)

    def test_uncertain_reservation_remains_owned_until_reconciled(self):
        with tempfile.TemporaryDirectory() as d:
            slot=Path(d)/'slot.json'
            slot.write_text(json.dumps(dict(state='uncertain',problem='muse_bf16_proj',experiment='P038')))
            with patch.object(coordinator,'GPU_SLOT',slot):
                with self.assertRaises(RuntimeError):
                    coordinator.gpu_slot('sending',dict(experiment='P039',submission_sha256='other'))
                coordinator.gpu_slot('terminal',dict(experiment='P038',submission_sha256='source',job_id='reconciled'))
            self.assertEqual(json.loads(slot.read_text())['state'],'terminal')


if __name__=='__main__':unittest.main(verbosity=2)

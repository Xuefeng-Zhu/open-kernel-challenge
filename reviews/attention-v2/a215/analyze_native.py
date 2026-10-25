"""Recompute the narrow native evidence audit from saved bytes; no GPU/network."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def main():
    receipt = json.loads((HERE / 'native-modal-run.json').read_text())
    folder = HERE / 'native_artifacts'
    manifest = receipt['transport_manifest']['artifact_sha256']
    assert set(manifest) == {p.name for p in folder.iterdir() if p.is_file()}
    for name, expected in manifest.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected, name
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['candidate_kernel_launches'] == 0
    assert summary['correctness_tested'] is summary['performance_tested'] is False
    records = summary['records']
    assert len(records) == 4 and all(r['sass_tool_exit_code'] == 0 for r in records)
    resources = [{k: r[k] for k in ('source', 'kernel', 'registers', 'driver_spills_raw', 'shared_memory_bytes')}
                 for r in records]
    def text(name):
        return (folder / name).read_text()
    sass = text('a215.producer.sass')
    instructions = {}
    for line_number, line in enumerate(sass.splitlines(), 1):
        match = re.search(r'/\*([0-9a-f]{4,})\*/\s+(.+?)\s*/\*', line)
        if match:
            instructions[int(match[1], 16)] = (line_number, match[2])
    pairs = [('R49', 0x2db0, 0x35b0), ('R48', 0x2de0, 0x35d0),
             ('R47', 0x2e30, 0x35f0), ('R46', 0x2e50, 0x3630),
             ('R89', 0x2f10, 0x3600), ('R62', 0x2f60, 0x3660)]
    uses = []
    for reg, load, use in pairs:
        ll, li = instructions[load]
        ul, ui = instructions[use]
        assert f'LDG.E {reg},' in li and 'VIMNMX' in ui
        # No intervening textual use/overwrite of this exact scalar register.
        assert not any(re.search(r'\b' + reg + r'\b', ins)
                       for addr, (_, ins) in instructions.items() if load < addr < use)
        assert re.search(r'\b' + reg + r'\b', ui)
        uses.append(dict(register=reg, load=hex(load), first_use=hex(use),
                         load_line=ll, use_line=ul, static_address_slots=(use-load)//16))
    assert text('a213.merge.sass') == text('a215.merge.sass')
    lowering = {}
    for version in ['a213', 'a215']:
        ptx = text(version + '.producer.ptx')
        mma = [line.strip() for line in ptx.splitlines()
               if re.search(r'\b(?:mma\.sync|wgmma\.)', line)]
        assert mma and any('.f16.f16.' in line for line in mma)
        assert not any('wgmma.' in line or '.e4m3' in line or '.e5m2' in line for line in mma)
        lowering[version] = dict(mma_instruction_families=sorted(set(line.split('{')[0].strip() for line in mma)),
                                 native_fp8_mma_present=False, wgmma_present=False)
    a13, a15 = resources[0], resources[2]
    report = dict(kind='Offline audit of saved native compilation; not runtime evidence',
                  app_id=receipt['app_id'], source_commit=receipt['source_commit'],
                  files_verified=len(manifest), artifact_bytes=sum((folder/n).stat().st_size for n in manifest),
                  resources=resources, shared_memory_delta_bytes=a15['shared_memory_bytes']-a13['shared_memory_bytes'],
                  page_load_use=uses, merge_sass_equal=True, lowering=lowering,
                  native_schedule_change=True, original_before_qk_order_preserved=False,
                  candidate_kernel_launches=0, correctness_tested=False, performance_tested=False)
    (HERE / 'native-analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(artifact_files_verified=len(manifest), load_use_pairs_verified=len(uses),
                         merge_sass_equal=True, native_fp8_mma=False, runtime_claims=False)))


if __name__ == '__main__':
    main()

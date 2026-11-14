#!/usr/bin/env python3
"""Measure an already-visible binocular preview over USB; restore power settings on exit."""

import argparse
import json
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--serial')
    parser.add_argument('--output', type=Path, default=Path('app/build/reports/preview-soak.json'))
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval < 15:
        parser.error('seconds must be positive; interval must be at least 15 seconds')
    command = ['adb'] + (['-s', args.serial] if args.serial else [])

    def adb(*parts):
        return subprocess.run(command + list(parts), check=True, capture_output=True,
                              text=True, timeout=20).stdout

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    remote_ui = '/data/local/tmp/memory-preview-soak.xml'
    try:
        adb('shell', 'uiautomator', 'dump', remote_ui)
        ui = ET.fromstring(adb('shell', 'cat', remote_ui))
        controls = ui.findall(".//node[@resource-id='org.cuxr.memory:id/controls']")
        if len(controls) != 2 or any('hide preview' not in node.get('text', '') for node in controls):
            raise RuntimeError('Enable binocular preview on page 4 before running the soak')
    finally:
        adb('shell', 'rm', '-f', remote_ui)
    original = adb('shell', 'settings', 'get', 'global', 'stay_on_while_plugged_in').strip()
    initial_pid = adb('shell', 'pidof', 'org.cuxr.memory').strip()
    report = {'seconds_requested': args.seconds, 'samples': [], 'passed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch, open(Path(scratch) / 'camera.log', 'w') as log_output:
        log_file = Path(scratch) / 'camera.log'
        stream = subprocess.Popen(command + ['logcat', '-T', '1', '-v', 'brief',
                                  'MemoryCamera:I', '*:S'], stdout=log_output, stderr=subprocess.DEVNULL)
        try:
            # Development-only USB keep-awake; never change the app's sleep behavior.
            adb('shell', 'svc', 'power', 'stayon', 'usb')
            adb('shell', 'dumpsys', 'gfxinfo', 'org.cuxr.memory', 'reset')
            start = time.monotonic()
            previous_frames = None
            previous_elapsed = 0.0
            while True:
                if adb('shell', 'pidof', 'org.cuxr.memory').strip() != initial_pid:
                    raise RuntimeError('App process changed')
                activities = adb('shell', 'dumpsys', 'activity', 'activities')
                if not re.search(r'topResumedActivity=.*org\.cuxr\.memory/\.HudActivity', activities):
                    raise RuntimeError('HUD is no longer resumed')
                memory = adb('shell', 'dumpsys', 'meminfo', 'org.cuxr.memory')
                battery = adb('shell', 'dumpsys', 'battery')
                thermal = adb('shell', 'dumpsys', 'thermalservice')
                stats = re.findall(r'Preview stats: frames=(\d+) sensorFps=([\d.]+) maxGapMs=(\d+)', log_file.read_text())
                frames = int(stats[-1][0]) if stats else None
                elapsed = time.monotonic() - start
                stalled = (elapsed - previous_elapsed >= 10 and previous_frames is not None
                           and frames is not None and frames <= previous_frames)
                if (elapsed > 10 and frames is None) or stalled:
                    raise RuntimeError('No new preview frame statistics; do not capture or change pages during the test')
                previous_frames = frames
                previous_elapsed = elapsed
                sample = {
                    'elapsed_s': round(elapsed, 1),
                    'pss_kb': int(re.search(r'^\s*TOTAL\s+(\d+)', memory, re.M)[1]),
                    'battery_percent': int(re.search(r'level: (\d+)', battery)[1]),
                    'battery_celsius': int(re.search(r'temperature: (\d+)', battery)[1]) / 10,
                    'thermal_status': int(re.search(r'Thermal Status: (\d+)', thermal)[1]),
                    'frames': frames,
                    'sensor_fps': float(stats[-1][1]) if stats else None,
                    'max_sensor_gap_ms': int(stats[-1][2]) if stats else None,
                }
                report['samples'].append(sample)
                print(json.dumps(sample), flush=True)
                if elapsed >= args.seconds:
                    report['gfxinfo'] = adb('shell', 'dumpsys', 'gfxinfo', 'org.cuxr.memory')
                    report['passed'] = True
                    break
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                time.sleep(min(args.interval, args.seconds - elapsed))
        finally:
            stream.terminate()
            try:
                stream.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stream.kill()
                stream.wait()
            if original == 'null':
                adb('shell', 'settings', 'delete', 'global', 'stay_on_while_plugged_in')
            else:
                adb('shell', 'settings', 'put', 'global', 'stay_on_while_plugged_in', original)
            args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

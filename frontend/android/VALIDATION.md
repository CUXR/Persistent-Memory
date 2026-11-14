# RayNeo camera validation — 2026-10-18

Issue: [#39](https://github.com/CUXR/Persistent-Memory/issues/39). Branch: `feat/rayneo-android-hud`.

## Device and build

- USB ADB authorized: RayNeoX3Pro / ARGF20 / MercuryLiteXR, Android 12/API 32, firmware `SKQ1.250204.001 release-keys`.
- JDK 21; `./gradlew :app:assembleDebug :app:lintDebug` passes. Lint: zero errors, 13 warnings (dependency/platform versions, SDK reflection resources, platform recommendations).
- Debug APK installed on the glasses. SHA-256: `1e4f01b5d31a89803fc3c4bc66accfa4170d0e093d43a65040f12923acbd73dc`. No backend connection or network permission added.

## Observed camera behavior

- Enumerated both cameras. Selected rear/backward-compatible camera 0 by sensor pixel area; camera 1 remains the smaller spatial camera.
- Advertised RGB sizes include 1920x1080 for SurfaceTexture and JPEG; AE ranges are 15–15, 24–24, 15–30, 30–30. Requests leave FPS at the platform default.
- A single 1920x1080 TextureView output produced completed preview frames. First capture then configured preview + 1920x1080 JPEG. Both sessions succeeded on this firmware.
- First JPEG: `1640bcd9-ebf0-4ba8-9178-552c3dc54bb4.jpg`, 1,654,625 bytes, decoded dimensions 1920x1080. Extracted with `adb exec-out run-as`; Mac image inspection shows the actual room/table. EXIF-aware viewing displays the scene upright with the requested sensor rotation (90 degrees). Raw encoded dimensions are landscape.
- Both eyes show the same camera title, capture result, pagination, and controls in a device screenshot. Wearer optical fusion remains unverified.
- Ten Home → resume → preview frame → capture cycles passed. Each cycle saved a JPEG. App-private retention held at ten files, pruning the oldest image. A second ten-cycle run against the final capture APK also passed (20 total cycles before adding the preview toggle).
- Runtime permission prompt rendered in both eyes. Revoking CAMERA killed the app as expected; relaunch without permission left no active camera clients. Regranting restored preview/capture. The vendor prompt requires physical gestures; denial-result copy and permanent-denial recovery still need a controlled wearer test.
- Camera service PID remained 889 across testing; no AndroidRuntime or native crash-buffer entries observed. Vendor stabilization warnings occur during streaming. Camera-service client-death events during permission revocation are not treated as HAL crashes.

## Reproduction

Use the build/install instructions and arrow/Enter controls in [README.md](README.md). Grant CAMERA through the glasses' prompt or, for development only:

```sh
adb shell pm grant org.cuxr.memory android.permission.CAMERA
```

Keep a filtered logcat running before exercising camera sessions; vendor logs can roll the device buffer quickly. Confirm `Preview frame received; JPEG output=false`, then `JPEG output=true` and `Saved ...` entries. For each lifecycle cycle, press Home, relaunch `org.cuxr.memory/.HudActivity`, wait for a new preview frame, press Enter, and wait for a new saved JPEG. Inspect `files/captures/` through `run-as`; scene images are intentionally excluded from Git.

## Additional capture checks

- Three rapid Enter actions while capture was pending produced one JPEG.
- Pausing during an in-flight capture released all active camera clients; resume recovered preview and another JPEG. Returning to a text page also left no active camera clients.
- OEM launcher may force-stop background apps after Home. Process exit history identifies launcher PID 1829 and `USER REQUESTED`, not an app/native crash. A fresh process starts on page one; immediate resumes retained page four.

## Binocular live preview

- Hold/P toggles the source TextureView in the left eye and Mercury MirroringView in the right. Capture remains available; hiding preview restores the text layout. Pause/page exit turns mirroring off.
- Device screenshots show the same upright room image with matching margins and status. A pixel comparison of the body regions differed by less than 0.0001 per RGB channel on the 0–255 scale. A JPEG captured with visible preview decoded successfully.
- TextureView already applies sensor rotation. Applying it again rotated the preview incorrectly; the final implementation only corrects aspect ratio. Reference: [Android camera preview](https://developer.android.com/media/camera/camera2/camera-preview#textureview).
- Ten-minute USB-powered soak passed (600.9 seconds, 21 samples), without app/camera-service/provider restart. Camera sensor-result rate stayed approximately 30 fps; longest observed sensor timestamp gap was 66 ms. This is not measured display latency or a count of dropped analysis frames.
- PSS ranged 49,285–59,429 KiB (initial warm-up high), ending at 50,696 KiB. No sustained memory growth appeared in this run. Thermal status remained 0; battery temperature fell from 29 to 26 C while USB charging (98% → 100%). Battery drain was not measurable under these conditions.
- Android gfxinfo recorded 23,633 rendered frames, 20 janky frames (0.08%), median 5 ms, 99th percentile 10 ms. It also recorded long-bin outliers and high-input-latency counters; end-to-end display latency remains unverified.
- The USB keep-awake setting was restored to its original value, 0. Raw samples and gfxinfo are in ignored `app/build/reports/preview-soak-2026-10-18.json` and `preview-gfx-2026-10-18.txt`.
- Final checks passed: hide/show toggle, preview remaining hidden after Home/resume, JPEG capture after re-enabling preview, and camera release on returning to a text page. Original USB sleep setting confirmed restored to 0.
- The reusable `tools/preview_soak.py` also passed a 16-second device smoke run, including its preview-state precheck, fresh frame-statistics collection, output report, and setting restoration.
- Optical fusion and the physical hold gesture still need wearer confirmation.

## Remaining acceptance

- Controlled physical permission deny/retry and persistent-denial Settings recovery.
- Physical temple gestures, optical fusion/readability, removal/wear behavior.
- Deliberate camera contention/disconnection.
- Raw YUV delivery, measured display latency, and discharge-only battery measurements remain outstanding.

## Android emulator checks (2026-11-14)

- Matching briefing text in both panes, all three modes, mode persistence after force-stop, fact wrapping, and a fact from the continuation fixture. Switching fixtures reset pagination and cleared the previous person's facts.
- Microphone permission denial displayed recovery guidance in both panes. After granting permission, tap-to-stop, leaving the page, and pressing Home each produced a distinct, decodable 16 kHz mono PCM16 WAV.
- Debug build and lint passed. RayNeo optics, temple inputs, microphone, and standalone Wi-Fi still need hardware validation.

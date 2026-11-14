# RayNeo X3 Pro Android frontend

Tracking issue: [#39 — RayNeo Android frontend](https://github.com/CUXR/Persistent-Memory/issues/39). The issue records implementation status, device evidence, camera diagnostics, and the staged backend integration work.

Build a standalone Kotlin APK for camera access and binocular text display. Keep backend integration out of this phase.

## Connectivity

X3 Pro has onboard Wi-Fi 6 and supports standalone operation. A phone can assist with initial setup or provide a hotspot; the app should run without an active phone connection. Verify direct networking with the phone disconnected before later backend integration.

Source: https://www.rayneo.com/products/x3-pro-ai-display-glasses

## 1. Device and development setup

- Install Android Studio and Android SDK Platform Tools on macOS.
- Obtain Mercury Android SDK and its matching Android sample through https://open.rayneo.com/.
- Enable developer mode on the glasses and connect using a USB-C data cable.
- Verify ADB authorization; inspect Android/API version, display geometry, camera capabilities, and firmware.
- Use Gradle Kotlin DSL; pin toolchain and dependency versions compatible with the actual SDK and device.
- Follow the matching SDK sample for application initialization and native-glasses manifest metadata.

## 2. Text display

- Use Kotlin, XML Views, and ViewBinding with Mercury's BaseMirrorActivity and BindingPair.
- Render identical content in both eyes within optical safe margins.
- Use large text, short lines, and a minimal HUD with status and sample memory text.
- Map temple gestures to cycle text and dismiss the app.
- Keep UI updates separate from state changes: paired rendering executes for both eyes.

Source: https://github.com/RayNeo-AI-2025/OpenClaw/blob/main/rayneo-x3-ar-dev-guide/03-core-concepts/01-dual-screen-rendering.md

## 3. Camera access

- Use Camera2 with runtime CAMERA permission.
- Enumerate camera IDs, stream sizes, frame rates, and supported output combinations; verify the main RGB camera rather than assuming an ID.
- Capture JPEGs into app-private storage and display capture status.
- Configure preview and ImageReader output surfaces in the capture session before targeting them in requests.
- Handle permission denial, camera contention, disconnection, lifecycle changes, and resource release.

Sources:
- https://github.com/RayNeo-AI-2025/OpenClaw/blob/main/rayneo-x3-ar-dev-guide/05-hardware-apis/01-camera.md
- https://developer.android.com/media/camera/camera2/capture-sessions-requests

## 4. Optional live camera mode

- Add preview with TextureView and Mercury MirroringView.
- Start around 720p at 15–30 fps if supported; measure latency, heat, and battery use.
- Add a bounded ImageReader frame stream for future processing. Drop stale frames and close every acquired image.
- Keep the text HUD over the wearer's natural view as the default. Make live preview a diagnostic mode.

## 5. Local demo and validation

- One screen: text HUD, preview toggle, and capture action.
- Small components: CameraController, HudViewModel, and TempleInputAdapter. Use local sample text.
- Verify readable fused binocular text, temple input, successful JPEG capture, permission denial/recovery, and repeated pause/resume camera reopening.
- Run a sustained preview session and record thermal/battery behavior.
- Verify app operation with the phone disconnected. Separately verify direct Wi-Fi networking without backend integration.
- Deliver source, installable APK, setup instructions, and actual device test results.

Hardware acceptance requires an actual X3 Pro; emulator/build success alone is insufficient.

## Verified setup (2026-10-03)

- USB ADB connection authorized; device identifies as RayNeoX3Pro / ARGF20 / MercuryLiteXR.
- Android 12, API 32; build SKQ1.250204.001 release-keys.
- Combined logical display: 1280×480, density 160 dpi.
- Camera service exposes IDs 0 and 1; no active camera clients during inspection.
- Camera 0 reports 15–30 fps ranges, fixed focus, and 90-degree sensor orientation. Confirm frame rotation and output combinations in the app.
- macOS tooling installed: Android Studio, ADB, Android command-line tools. SDK root: /Users/engai/Library/Android/sdk.
- Mercury v0.2.5 AAR and sample cached in /Users/engai/Library/Caches/rayneo-x3-sdk; SOURCE.txt records the source commit and SHA-256 hashes.
- Direct Wi-Fi independence and camera capture remain to be tested on this device.

## Text milestone (2026-10-03)

- Implemented Kotlin/Gradle app in this folder; SDK initialization and native-glasses manifest metadata match the vendor sample.
- Three local text pages rendered with BaseMirrorActivity and paired ViewBinding; temple tap/swipes browse, double-tap exits.
- Page state uses SavedStateHandle; consumed gesture events are not replayed after resume.
- Debug build and Android lint pass (warnings remain for dependency versions, SDK reflection-based resources, and platform recommendations).
- APK installed and launched on X3 Pro; device screenshot confirms matching left/right text. Relaunch succeeded without app crashes.
- Wearer readability and physical gesture behavior need wearer confirmation. Ordinary ADB touchscreen injection does not emulate the SDK's right-temple input device.
- Camera capture and preview remain the next milestone; direct scrcpy camera capture failed in the vendor camera HAL during stream configuration.

## Camera milestone (2026-10-18)

- Implemented permission flow, characteristic-based RGB selection, single-surface diagnostic preview, then JPEG session configuration on first capture.
- Added camera page and single-dispatch TempleInputAdapter; both eyes receive the same status. Arrow/Enter controls support ADB diagnostics.
- Camera lifetime owns worker, capture timeout, session/device/reader/surface cleanup; pause and page exit release camera resources. Removed permanently awake window flag.
- Hardware delivers 1920x1080 preview and JPEG on camera 0. Mac extraction confirms real scene content and EXIF rotation. Ten home/resume/capture cycles succeeded; retention stays at ten JPEGs.
- Build/lint pass. Detailed evidence and unverified permission/physical-input/contended-camera checks: [VALIDATION.md](VALIDATION.md).
- Next: finish remaining wearer/permission checks; binocular preview is implemented below. Backend wiring remains gated by issue #39.

## Binocular preview (2026-10-18)

- Hold on Local camera to toggle live preview; tap captures. P provides the same toggle through ADB/keyboard.
- One TextureView output feeds the left eye; Mercury MirroringView redraws that texture in the right eye. No second camera stream or app-level frame queue is added.
- Fit the sensor-oriented texture within the body region. Status uses the existing small text token on black; text-only capture remains the default and returns on pause/page exit.
- Device screenshots confirm matching upright scene content in both eyes; JPEG capture still succeeds while preview is visible. Long-run results are recorded in VALIDATION.md.
- Raw YUV delivery, display latency measurement, standalone Wi-Fi, and backend integration remain separate work.

## Proposed local-server POC after device checks

The current checkout's FastAPI app mounts an empty `/users` router. There is no recognition/upload/briefing HTTP contract to consume yet.

1. Add an explicitly development-only, loopback-bound receiver with a health route, bounded JPEG upload, and deterministic fixture briefing response. Propose routes under `/dev/device/`; agree schemas before wiring real recognition. Do not represent fixture identities as persisted people.
2. Forward glasses loopback to the Mac with `adb reverse tcp:8000 tcp:8000`. A debug Kotlin client can then reach `http://127.0.0.1:8000`. Keep cleartext exceptions restricted to debug and loopback; production needs HTTPS and authenticated device provisioning.
3. Start with manual captured-JPEG submission, one request in flight, a fixed upload limit (proposed 4 MiB), timeout/cancellation, session ID, sequence, capture timestamp, and explicit EXIF orientation semantics. Do not add continuous streaming until backpressure and the transport contract are proven.
4. Render bounded fixture text plus connecting/received/offline/error states on the HUD. Verify actual upload decoding on the Mac, unplug/reconnect recovery, and stale-response cancellation on page/session changes.
5. Replace fixtures only after #34/#35 and the authenticated recognition/briefing contracts are available. Preserve owner isolation and persisted person UUIDs; add audio boundaries before microphone integration.

This is a proposed next slice, not an implemented API or a production authentication design.

## Fixture HUD and microphone diagnostic (2026-11-14)

- The briefing page uses PR #52 fixtures for locally persisted hidden, name, and name + facts modes. Emulator checks cover identity changes and fact pagination.
- The microphone page records a bounded app-private WAV for hardware qualification. Device audio intake and the upload format are tracked in #49 and #50.
- Next hardware checks: wearer input and readability, microphone quality and permissions, and direct Wi-Fi. Follow the scope amendment in #39 for backend integration.

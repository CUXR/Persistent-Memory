# Memory glasses

Standalone RayNeo X3 Pro HUD with fixture briefings, local Camera2 capture, and a microphone diagnostic. Swipe to browse five pages and double-tap to exit. Camera and microphone permissions are requested on demand. No network access.

## Build

Use JDK 21, Android SDK platform 36, and the checked-in Gradle wrapper. Configure Android Studio's Gradle JDK to 21 (the current Studio bundled JDK 25 is too new for Gradle 8.13).

Create ignored `local.properties` with your SDK path:

```properties
sdk.dir=/Users/engai/Library/Android/sdk
```

Place `MercuryAndroidSDK-v0.2.5-20260212110627_ceaebc13.aar` in `app/libs/`. This Mac already has a copy in `/Users/engai/Library/Caches/rayneo-x3-sdk/`.

SDK source: [RayNeo repository](https://github.com/RayNeo-AI-2025/OpenClaw/blob/6802e8d8dfd63f087447f9ef7a7135ebe69e48c4/rayneo-x3-ar-dev-guide/assets/MercuryAndroidSDK-v0.2.5-20260212110627_ceaebc13.aar). SHA-256: `dda8df37b26818f27e5e081c0d22860ad02370ff4a84087c7a45f800384a9564`. Vendor SDK licensing applies; the binary is excluded from Git.

```sh
export JAVA_HOME=$(/usr/libexec/java_home -v 21)
./gradlew :app:assembleDebug :app:lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n org.cuxr.memory/.HudActivity
```

Compile and target SDK are 36; minimum API is 32, supporting the glasses' Android 12 firmware. Text uses Mercury's paired ViewBinding rendering, and page state survives recreation through SavedStateHandle.

## Device checks

- Both eyes show matching text; wearing the glasses produces one readable image.
- Tap and both swipe directions change exactly one page; pages wrap.
- Double-tap exits to the launcher.
- Returning after pausing preserves the page without replaying the last gesture. Double-tap finishes the activity; a fresh launch starts on page one.
- Check `adb logcat -s MemoryHud AndroidRuntime` for page renders or crashes.

## Fixture briefings

Swipe once from Ready. Hold to cycle Name + facts → Hidden → Name. The mode survives relaunch. Tap to advance one fact, including facts from the bundled continuation page. In Hidden mode, hold again to restore text or swipe to leave the page.

For ADB or keyboard checks, M cycles modes and F switches between the named, no-facts, and unnamed fixtures. The bundled responses come from [PR #52](https://github.com/CUXR/Persistent-Memory/pull/52). The HUD displays hobbies and affiliations and omits visual descriptors.

## Local microphone

Swipe to page 5 and tap to request permission. Tap again to record and tap to stop. Leaving the page or app also stops recording. The diagnostic records for up to 60 seconds and keeps the latest successful sample as app-private 16 kHz mono PCM16 WAV.

```sh
adb exec-out run-as org.cuxr.memory cat files/microphone.wav > /tmp/memory-microphone.wav
```

## Local camera

Browse to page 4, **Local camera**. Tap to request CAMERA permission; tap again after "Camera active" to capture. The camera runs only while this page is resumed. Leaving the page, sleeping, or exiting closes it. Screen sleep remains enabled.

A single TextureView supplies the diagnostic preview. Hold on the camera page to show/hide it in both eyes; text-only capture remains the default. Mercury MirroringView redraws the source texture in the other eye, without a second camera output. Aspect-ratio fitting accounts for TextureView's built-in sensor rotation. Preview hides on pause or page exit. After its first completed frame, the first capture adds a JPEG ImageReader to the session; subsequent taps reuse that session. Camera selection uses rear-facing, backward-compatible characteristics and sensor area, rather than a fixed ID. Logs include advertised stream sizes, FPS, sensor orientation, and selected configuration. No unsupported FPS override is applied.

JPEGs stay in app-private `files/captures/`; only the latest ten are retained. Rotation is stored using the sensor orientation for the glasses' fixed natural landscape display. Extract a named capture for Mac inspection:

```sh
adb shell run-as org.cuxr.memory ls files/captures
adb exec-out run-as org.cuxr.memory cat files/captures/NAME.jpg > /tmp/memory-capture.jpg
adb logcat -v threadtime MemoryCamera:I MemoryHud:I AndroidRuntime:E '*:S'
```

For USB keyboard/ADB diagnostics, left/right arrows browse, Enter/DPAD-center performs the tap action, and P toggles preview. This tests app behavior; physical temple input and optical fusion still require wearer checks. For example, from a fresh launch:

```sh
adb shell input keyevent KEYCODE_WAKEUP
adb shell wm dismiss-keyguard
adb shell input keyevent 22
adb shell input keyevent 22
adb shell input keyevent 22
adb shell input keyevent 66
```

On errors, swipe away and return to reopen. Permission denial shows a retry hint; persistent denial requires Android Settings. Each camera lifetime has a dedicated worker, bounded capture timeout, and at most one pending capture. Stale callbacks cannot repaint another page or resumed lifetime.

See [VALIDATION.md](VALIDATION.md) for measured device results and remaining acceptance checks. Sustained preview and direct Wi-Fi work are tracked in [PLAN.md](PLAN.md).

## Preview diagnostics

With page 4 visible and preview enabled, run from this directory:

```sh
python3 tools/preview_soak.py --seconds 600
```

The standard-library-only tool records process continuity, preview sensor-result FPS/gaps, PSS, battery temperature/charge, thermal status, and Android rendering statistics into ignored `app/build/reports/preview-soak.json`. It temporarily enables USB keep-awake and restores the prior setting on exit. Keep the glasses connected and avoid capture/page changes during the run. Sensor-result FPS is not display latency; USB charging cannot establish battery drain. The app itself keeps normal sleep behavior.

A proposed ADB-to-local-server POC is outlined in [PLAN.md](PLAN.md); no server transport is implemented yet.

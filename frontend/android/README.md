# Memory glasses

Standalone RayNeo X3 Pro text HUD. Three local pages; tap or swipe to browse, double-tap to exit. No network or camera permissions in this milestone.

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

Camera capture, optional preview, and direct Wi-Fi verification remain in [PLAN.md](PLAN.md).

package org.cuxr.memory

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.os.Bundle
import android.util.Log
import android.view.KeyEvent
import android.view.TextureView
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.ffalcon.mercury.android.sdk.ui.activity.BaseMirrorActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.cuxr.memory.databinding.ActivityHudBinding

class HudActivity : BaseMirrorActivity<ActivityHudBinding>() {
    private lateinit var hud: HudViewModel
    private var camera: CameraController? = null
    private var resumed = false
    private lateinit var preview: BinocularPreview
    private val displayPreferences by lazy { getSharedPreferences("hud_display", MODE_PRIVATE) }
    private var briefingMode = BriefingMode.FACTS
    private val fixtures by lazy { BriefingFixtures(assets) }
    private lateinit var briefing: Briefing
    private var renderedPage = -1
    private var cameraMessage = "Tap to allow camera access."
    private var microphone: MicrophoneCapture? = null
    private var microphoneMessage = R.string.microphone_ready
    private val microphonePermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        microphoneMessage = if (granted) R.string.microphone_ready else R.string.microphone_denied
        if (resumed && hud.page.value == HudViewModel.MICROPHONE_PAGE) render(hud.page.value)
    }
    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        cameraMessage = if (granted) "Starting camera..." else "Camera denied. Tap to retry.\nEnable in Settings if blocked."
        if (resumed && hud.page.value == HudViewModel.CAMERA_PAGE) {
            if (granted) startCamera()
            render(hud.page.value)
        }
    }
    private val input = TempleInputAdapter(
        next = { hud.advance(it) },
        tap = ::onTap,
        exit = { finish() },
        hold = ::onHold,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        hud = ViewModelProvider(this)[HudViewModel::class.java]
        val savedMode = displayPreferences.getString("briefing_mode", null)
        briefingMode = BriefingMode.entries.firstOrNull { it.name == savedMode } ?: BriefingMode.FACTS
        briefing = fixtures.load(hud.fixtureIndex)
        preview = BinocularPreview(mBindingPair)
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.RESUMED) {
                launch { hud.page.collect(::render) }
                launch {
                    templeActionViewModel.state.collect { action ->
                        input.accept(action)
                    }
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        resumed = true
        if (hud.page.value == HudViewModel.CAMERA_PAGE) startCamera()
    }

    override fun onPause() {
        resumed = false
        preview.setEnabled(false)
        stopCamera()
        stopMicrophone()
        super.onPause()
    }

    private fun onTap() {
        when (hud.page.value) {
            HudViewModel.BRIEFING_PAGE -> {
                if (briefingMode == BriefingMode.FACTS && briefing.facts.isNotEmpty()) {
                    hud.factIndex = (hud.factIndex + 1) % briefing.facts.size
                    render(hud.page.value)
                    return
                }
            }
            HudViewModel.MICROPHONE_PAGE -> {
                if (microphone == null) startMicrophone() else stopMicrophone()
                return
            }
            HudViewModel.CAMERA_PAGE -> {
                if (!hasCameraPermission()) permission.launch(Manifest.permission.CAMERA)
                else if (camera == null) startCamera() else camera?.capture()
                return
            }
        }
        hud.advance(1)
    }

    private fun onHold() {
        when (hud.page.value) {
            HudViewModel.BRIEFING_PAGE -> {
                briefingMode = briefingMode.next()
                displayPreferences.edit { putString("briefing_mode", briefingMode.name) }
                render(hud.page.value)
            }
            HudViewModel.CAMERA_PAGE -> togglePreview()
        }
    }

    private fun nextFixture() {
        if (hud.page.value != HudViewModel.BRIEFING_PAGE) return
        hud.fixtureIndex = (hud.fixtureIndex + 1) % fixtures.count
        hud.factIndex = 0
        briefing = fixtures.load(hud.fixtureIndex)
        render(hud.page.value)
    }

    private fun startMicrophone() {
        if (!resumed) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            microphonePermission.launch(Manifest.permission.RECORD_AUDIO)
            return
        }
        val capture = MicrophoneCapture(filesDir)
        microphone = capture
        microphoneMessage = R.string.microphone_starting
        render(hud.page.value)
        lifecycleScope.launch {
            microphoneMessage = withContext(Dispatchers.IO) {
                try {
                    val samples = capture.record {
                        runOnUiThread {
                            if (resumed && microphone === capture && microphoneMessage == R.string.microphone_starting) {
                                microphoneMessage = R.string.microphone_recording
                                if (hud.page.value == HudViewModel.MICROPHONE_PAGE) render(hud.page.value)
                            }
                        }
                    }
                    if (samples > 0) R.string.microphone_saved else R.string.microphone_ready
                } catch (error: Exception) {
                    Log.w("MemoryHud", "Microphone capture failed", error)
                    R.string.microphone_failed
                }
            }
            microphone = null
            if (resumed && hud.page.value == HudViewModel.MICROPHONE_PAGE) render(hud.page.value)
        }
    }

    private fun stopMicrophone() {
        val capture = microphone ?: return
        capture.stop()
        microphoneMessage = R.string.microphone_stopping
        if (resumed && hud.page.value == HudViewModel.MICROPHONE_PAGE) render(hud.page.value)
    }

    private fun hasCameraPermission() = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        if (!resumed || camera != null) return
        if (!hasCameraPermission()) return
        val sink = preview.source
        sink.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(texture: SurfaceTexture, width: Int, height: Int) {
                if (resumed && hud.page.value == HudViewModel.CAMERA_PAGE) openCamera(texture)
            }
            override fun onSurfaceTextureDestroyed(texture: SurfaceTexture): Boolean {
                stopCamera()
                return true
            }
            override fun onSurfaceTextureSizeChanged(texture: SurfaceTexture, width: Int, height: Int) = preview.transform()
            override fun onSurfaceTextureUpdated(texture: SurfaceTexture) = Unit
        }
        if (sink.isAvailable) sink.surfaceTexture?.let(::openCamera)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean = when (keyCode) {
        KeyEvent.KEYCODE_M -> { if (hud.page.value == HudViewModel.BRIEFING_PAGE) onHold(); true }
        KeyEvent.KEYCODE_F -> { nextFixture(); true }
        KeyEvent.KEYCODE_P -> { togglePreview(); true }
        KeyEvent.KEYCODE_DPAD_RIGHT -> { hud.advance(1); true }
        KeyEvent.KEYCODE_DPAD_LEFT -> { hud.advance(-1); true }
        KeyEvent.KEYCODE_ENTER, KeyEvent.KEYCODE_DPAD_CENTER -> { onTap(); true }
        else -> super.onKeyUp(keyCode, event)
    }

    private fun openCamera(texture: SurfaceTexture) {
        if (camera != null) return
        cameraMessage = "Starting camera..."
        lateinit var owner: CameraController
        owner = CameraController(this, { size, orientation ->
            runOnUiThread { if (camera === owner) preview.configure(size, orientation) }
        }) { message ->
            runOnUiThread {
                if (camera === owner && resumed && hud.page.value == HudViewModel.CAMERA_PAGE) {
                    cameraMessage = message
                    render(hud.page.value)
                }
            }
        }
        camera = owner
        owner.start(texture)
    }

    private fun togglePreview() {
        if (hud.page.value != HudViewModel.CAMERA_PAGE) return
        preview.setEnabled(!preview.enabled)
        Log.i("MemoryHud", "Binocular preview ${preview.enabled}")
        render(hud.page.value)
    }

    private fun stopCamera() {
        preview.setEnabled(false)
        camera?.close()
        camera = null
    }

    private fun render(page: Int) {
        if (renderedPage != page) {
            renderedPage = page
            if (page != HudViewModel.MICROPHONE_PAGE) stopMicrophone()
            if (page == HudViewModel.CAMERA_PAGE) startCamera() else stopCamera()
        }
        if (page == HudViewModel.BRIEFING_PAGE) {
            renderBriefing()
            return
        }
        val titles = resources.getStringArray(R.array.hud_titles)
        val messages = resources.getStringArray(R.array.hud_messages)
        mBindingPair.updateView {
            hudLabel.visibility = View.VISIBLE
            title.visibility = View.VISIBLE
            message.visibility = View.VISIBLE
            pageNumber.visibility = View.VISIBLE
            controls.visibility = View.VISIBLE
            title.text = titles[page]
            message.text = when (page) {
                HudViewModel.CAMERA_PAGE -> cameraMessage
                HudViewModel.MICROPHONE_PAGE -> getString(microphoneMessage)
                else -> messages[page]
            }
            controls.text = getString(when {
                page == HudViewModel.MICROPHONE_PAGE -> R.string.microphone_controls
                page != HudViewModel.CAMERA_PAGE -> R.string.hud_controls
                preview.enabled -> R.string.preview_controls
                else -> R.string.camera_controls
            })
            pageNumber.text = getString(R.string.page_number, page + 1, HudViewModel.PAGE_COUNT)
        }
        Log.i("MemoryHud", "Rendered page ${page + 1} in both eyes")
    }

    private fun renderBriefing() {
        val hidden = briefingMode == BriefingMode.HIDDEN
        val showFacts = briefingMode == BriefingMode.FACTS
        val fact = briefing.facts.getOrNull(hud.factIndex)
        val name = briefing.name ?: getString(R.string.briefing_unnamed)
        mBindingPair.updateView {
            hudLabel.visibility = if (hidden) View.GONE else View.VISIBLE
            title.visibility = if (hidden) View.GONE else View.VISIBLE
            message.visibility = if (showFacts) View.VISIBLE else View.GONE
            pageNumber.visibility = if (showFacts && fact != null) View.VISIBLE else View.GONE
            controls.visibility = if (hidden) View.GONE else View.VISIBLE
            title.text = if (hidden) "" else name
            message.text = if (showFacts) fact ?: getString(R.string.briefing_no_facts) else ""
            pageNumber.text = if (showFacts && fact != null) {
                getString(R.string.page_number, hud.factIndex + 1, briefing.facts.size)
            } else ""
            controls.text = getString(
                if (showFacts && fact != null) R.string.briefing_fact_controls else R.string.briefing_controls,
                getString(briefingMode.label),
            )
        }
    }
}

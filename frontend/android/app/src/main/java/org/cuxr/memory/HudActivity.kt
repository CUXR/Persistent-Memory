package org.cuxr.memory

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.os.Bundle
import android.util.Log
import android.view.KeyEvent
import android.view.TextureView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.ffalcon.mercury.android.sdk.ui.activity.BaseMirrorActivity
import kotlinx.coroutines.launch
import org.cuxr.memory.databinding.ActivityHudBinding

class HudActivity : BaseMirrorActivity<ActivityHudBinding>() {
    private lateinit var hud: HudViewModel
    private var camera: CameraController? = null
    private var resumed = false
    private lateinit var preview: BinocularPreview
    private var renderedPage = -1
    private var cameraMessage = "Tap to allow camera access."
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
        togglePreview = ::togglePreview,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        hud = ViewModelProvider(this)[HudViewModel::class.java]
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
        super.onPause()
    }

    private fun onTap() {
        if (hud.page.value != HudViewModel.CAMERA_PAGE) { hud.advance(1); return }
        if (!hasCameraPermission()) permission.launch(Manifest.permission.CAMERA)
        else if (camera == null) startCamera() else camera?.capture()
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
            if (page == HudViewModel.CAMERA_PAGE) startCamera() else stopCamera()
        }
        val titles = resources.getStringArray(R.array.hud_titles)
        val messages = resources.getStringArray(R.array.hud_messages)
        mBindingPair.updateView {
            title.text = titles[page]
            message.text = if (page == HudViewModel.CAMERA_PAGE) cameraMessage else messages[page]
            controls.text = getString(when {
                page != HudViewModel.CAMERA_PAGE -> R.string.hud_controls
                preview.enabled -> R.string.preview_controls
                else -> R.string.camera_controls
            })
            pageNumber.text = getString(R.string.page_number, page + 1, HudViewModel.PAGE_COUNT)
        }
        Log.i("MemoryHud", "Rendered page ${page + 1} in both eyes")
    }
}

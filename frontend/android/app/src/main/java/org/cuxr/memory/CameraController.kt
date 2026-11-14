package org.cuxr.memory

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import java.io.File
import java.util.UUID
import java.util.concurrent.Executor

/** One camera lifetime; all camera and file work runs on the worker. */
class CameraController(
    context: Context,
    private val configurePreview: (Size, Int) -> Unit,
    private val report: (String) -> Unit,
) {
    private val manager = context.getSystemService(CameraManager::class.java)
    private val directory = File(context.filesDir, "captures")
    private val thread = HandlerThread("MemoryCamera").apply { start() }
    private val worker = Handler(thread.looper)
    private val executor = Executor { worker.post(it) }
    @Volatile private var closed = false
    private var opening = false
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var preview: Surface? = null
    private var reader: ImageReader? = null
    private var characteristics: CameraCharacteristics? = null
    private var pendingCapture = false
    private var addingJpeg = false
    private val timeout = Runnable { fail("Camera timed out. Swipe away, then retry.") }

    @SuppressLint("MissingPermission")
    fun start(texture: SurfaceTexture) {
        worker.post {
            if (closed) return@post
            try {
                val cameras = manager.cameraIdList.map { it to manager.getCameraCharacteristics(it) }
                cameras.forEach { (id, info) ->
                    val map = info.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                    Log.i(TAG, "id=$id facing=${info.get(CameraCharacteristics.LENS_FACING)} " +
                        "capabilities=${info.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.contentToString()} " +
                        "orientation=${info.get(CameraCharacteristics.SENSOR_ORIENTATION)} " +
                        "preview=${map?.getOutputSizes(SurfaceTexture::class.java)?.contentToString()} " +
                        "jpeg=${map?.getOutputSizes(ImageFormat.JPEG)?.contentToString()} " +
                        "fps=${info.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)?.contentToString()}")
                }
                val selected = cameras.filter { (_, info) ->
                    info.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
                        ?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_BACKWARD_COMPATIBLE) == true &&
                        info.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
                }.maxByOrNull { (_, info) ->
                    info.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)?.let { it.width.toLong() * it.height } ?: 0L
                } ?: error("No rear RGB camera available")
                characteristics = selected.second
                val map = checkNotNull(selected.second.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP))
                val sizes = checkNotNull(map.getOutputSizes(SurfaceTexture::class.java))
                // Match Mercury's RGB preview size when advertised, without its unsupported fixed FPS override.
                val size = sizes.firstOrNull { it == Size(1920, 1080) }
                    ?: sizes.filter { it.width <= 1920 && it.height <= 1080 }.maxByOrNull { it.width * it.height }
                    ?: sizes.minBy { it.width * it.height }
                configurePreview(size, selected.second.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0)
                texture.setDefaultBufferSize(size.width, size.height)
                preview = Surface(texture)
                status("Opening camera ${selected.first}...")
                Log.i(TAG, "Selected RGB id=${selected.first}, preview=$size")
                opening = true
                worker.postDelayed(timeout, 10_000)
                manager.openCamera(selected.first, object : CameraDevice.StateCallback() {
                    override fun onOpened(camera: CameraDevice) {
                        opening = false
                        device = camera
                        if (closed) release() else guarded { configure(camera, false) }
                    }
                    override fun onDisconnected(camera: CameraDevice) {
                        opening = false
                        device = camera
                        fail("Camera disconnected. Swipe away, then retry.")
                    }
                    override fun onError(camera: CameraDevice, error: Int) {
                        opening = false
                        device = camera
                        fail("Camera error $error. Swipe away, then retry.")
                    }
                    override fun onClosed(camera: CameraDevice) {
                        if (closed) thread.quitSafely()
                    }
                }, worker)
            } catch (error: Exception) {
                opening = false
                Log.e(TAG, "Camera open failed", error)
                fail("Camera unavailable.\nSwipe away, then retry.")
            }
        }
    }

    private fun configure(camera: CameraDevice, withJpeg: Boolean) {
        session?.close()
        session = null
        val outputs = mutableListOf(OutputConfiguration(checkNotNull(preview)))
        if (withJpeg) {
            val sizes = checkNotNull(characteristics?.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?.getOutputSizes(ImageFormat.JPEG))
            val size = sizes.filter { it.width <= 1920 && it.height <= 1080 }.maxByOrNull { it.width * it.height }
                ?: sizes.minBy { it.width * it.height }
            reader = ImageReader.newInstance(size.width, size.height, ImageFormat.JPEG, 2).apply {
                setOnImageAvailableListener({ source -> saveImage(source) }, worker)
            }
            outputs.add(OutputConfiguration(checkNotNull(reader).surface))
            Log.i(TAG, "Configuring preview + JPEG=$size")
        }
        camera.createCaptureSession(SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outputs, executor,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(configured: CameraCaptureSession) {
                    if (closed) { configured.close(); return }
                    session = configured
                    guarded {
                        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                            addTarget(checkNotNull(preview))
                        }
                        configured.setRepeatingRequest(request.build(), object : CameraCaptureSession.CaptureCallback() {
                            private var firstFrame = true
                            private var frames = 0
                            private var firstTimestamp = 0L
                            private var previousTimestamp = 0L
                            private var longestGap = 0L
                            override fun onCaptureCompleted(s: CameraCaptureSession, r: CaptureRequest, result: TotalCaptureResult) {
                                if (closed) return
                                val timestamp = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: 0L
                                if (timestamp > 0L) {
                                    if (firstTimestamp == 0L) firstTimestamp = timestamp
                                    if (previousTimestamp > 0L) longestGap = maxOf(longestGap, timestamp - previousTimestamp)
                                    previousTimestamp = timestamp
                                    frames++
                                    if (frames % 150 == 0 && timestamp > firstTimestamp) {
                                        val fps = (frames - 1) * 1_000_000_000.0 / (timestamp - firstTimestamp)
                                        Log.i(TAG, "Preview stats: frames=$frames sensorFps=$fps maxGapMs=${longestGap / 1_000_000}")
                                    }
                                }
                                if (!firstFrame) return
                                firstFrame = false
                                worker.removeCallbacks(timeout)
                                Log.i(TAG, "Preview frame received; JPEG output=$withJpeg")
                                if (withJpeg) takeStill() else status("Camera active. Tap to capture.")
                            }
                        }, worker)
                    }
                }
                override fun onConfigureFailed(failed: CameraCaptureSession) {
                    failed.close()
                    fail("Camera session rejected. See MemoryCamera logs.")
                }
            }))
    }

    fun capture() {
        if (closed) return
        worker.post {
            if (closed || pendingCapture) return@post
            if (session == null) { status("Camera starting. Please wait."); return@post }
            pendingCapture = true
            status("Capturing...")
            worker.postDelayed(timeout, 10_000)
            guarded {
                if (!addingJpeg) {
                    addingJpeg = true
                    configure(checkNotNull(device), true)
                } else takeStill()
            }
        }
    }

    private fun takeStill() {
        if (!pendingCapture) return
        val info = checkNotNull(characteristics)
        // The HUD is locked to the glasses' natural landscape orientation.
        val orientation = info.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        val request = checkNotNull(device).createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(checkNotNull(reader).surface)
            set(CaptureRequest.JPEG_ORIENTATION, orientation)
        }
        checkNotNull(session).capture(request.build(), object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureFailed(s: CameraCaptureSession, r: CaptureRequest, failure: CaptureFailure) {
                fail("Capture failed: ${failure.reason}. Swipe away, then retry.")
            }
        }, worker)
    }

    private fun saveImage(source: ImageReader) = guarded {
        val image = source.acquireLatestImage() ?: return@guarded
        val bytes = image.use {
            val buffer = it.planes[0].buffer
            ByteArray(buffer.remaining()).also(buffer::get)
        }
        if (closed || !pendingCapture) return@guarded
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
        check(options.outWidth > 0 && options.outHeight > 0) { "Invalid JPEG" }
        check(directory.isDirectory || directory.mkdirs()) { "Cannot create capture directory" }
        // Bound local retention; scene images never leave app-private storage automatically.
        directory.listFiles()?.filter { it.extension == "jpg" }?.sortedByDescending { it.lastModified() }
            ?.drop(9)?.forEach { check(it.delete()) { "Cannot prune old capture" } }
        val target = File(directory, "${UUID.randomUUID()}.jpg")
        try { target.writeBytes(bytes) } catch (error: Exception) { target.delete(); throw error }
        pendingCapture = false
        worker.removeCallbacks(timeout)
        Log.i(TAG, "Saved ${target.name}: ${options.outWidth}x${options.outHeight}, ${bytes.size} bytes")
        status("Saved ${options.outWidth} x ${options.outHeight}.\nTap to capture again.")
    }

    private fun guarded(block: () -> Unit) {
        try { block() } catch (error: Exception) {
            Log.e(TAG, "Camera operation failed", error)
            fail("Camera failed.\nSwipe away, then retry.")
        }
    }

    private fun status(message: String) { if (!closed) report(message) }

    private fun fail(message: String) {
        if (!closed) {
            Log.e(TAG, message)
            status(message)
        }
        closed = true
        release()
    }

    fun close() {
        if (closed) return
        closed = true
        worker.post { release() }
    }

    private fun release() {
        worker.removeCallbacks(timeout)
        session?.close()
        session = null
        val closing = device
        device = null
        closing?.close()
        reader?.close()
        reader = null
        preview?.release()
        preview = null
        if (closing == null && !opening) thread.quitSafely()
    }

    companion object { private const val TAG = "MemoryCamera" }
}

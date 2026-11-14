package org.cuxr.memory

import android.graphics.Color
import android.graphics.Matrix
import android.util.Size
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import com.ffalcon.mercury.android.sdk.core.BindingPair
import org.cuxr.memory.databinding.ActivityHudBinding
import kotlin.math.min

/** One camera output; Mercury redraws the same transformed texture for the other eye. */
class BinocularPreview(private val bindings: BindingPair<ActivityHudBinding>) {
    val source = bindings.left.previewSink
    private val mirror = bindings.right.previewMirror
    private var size = Size(1920, 1080)
    private var orientation = 0
    var enabled = false
        private set

    init { mirror.setSource(source) }

    fun configure(size: Size, orientation: Int) {
        this.size = size
        this.orientation = orientation
        transform()
    }

    fun setEnabled(value: Boolean) {
        enabled = value
        source.alpha = if (value) 1f else 0f
        mirror.visibility = if (value) View.VISIBLE else View.GONE
        if (value) mirror.startMirroring() else mirror.stopMirroring()
        bindings.updateView {
            message.setTextSize(TypedValue.COMPLEX_UNIT_PX,
                root.resources.getDimension(if (value) R.dimen.hud_text_small else R.dimen.hud_text_body))
            message.layoutParams = (message.layoutParams as FrameLayout.LayoutParams).apply {
                gravity = if (value) Gravity.BOTTOM else Gravity.TOP
            }
            message.setBackgroundColor(if (value) root.resources.getColor(R.color.hud_background, null) else Color.TRANSPARENT)
        }
        source.post(::transform)
    }

    fun transform() {
        val width = source.width.toFloat()
        val height = source.height.toFloat()
        if (width <= 0 || height <= 0) return
        val rotated = orientation % 180 != 0
        val fittedWidth = if (rotated) size.height else size.width
        val fittedHeight = if (rotated) size.width else size.height
        val fit = min(width / fittedWidth, height / fittedHeight)
        // TextureView already applies sensor rotation; correct only its aspect-ratio stretch.
        source.setTransform(Matrix().apply {
            setScale(fittedWidth * fit / width, fittedHeight * fit / height, width / 2, height / 2)
        })
    }
}
